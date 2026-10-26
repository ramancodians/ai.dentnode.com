"""LiveKit voice agent: a real-time conversation on a clinic's behalf.

Runs as its own process, never inside the FastAPI server:

    python -m livekit.agents start voice_call/worker.py

It registers with LiveKit Cloud under ``AGENT_NAME`` and joins a room only when
dispatched there by name. LiveKit Agents supplies the parts the Voice Lab built
by hand, done properly: streaming STT, end-of-turn detection (Hindi included),
barge-in, and TTS that starts on the reply's first sentence.

The LLM is the one hop that leaves LiveKit. It goes to OpenRouter through the
OpenAI-compatible plugin, so it follows the platform rule (one OpenRouter key,
no provider SDK keys) and is metered into the usage ledger when the call ends.

Every finished message is also published to the room on ``DATA_TOPIC`` with
LiveKit's per-turn latency report, which is what the Voice Lab page renders.
"""

import asyncio
import json
import logging
import os
import time

from livekit import rtc

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, inference
from livekit.agents.worker import ServerEnvOption
from livekit.plugins import openai as lk_openai

from agent.calling_prompt import calling_system_prompt
from agent.config import settings
from agent.voice_agent import _REASONING_MANDATORY

from .config import AGENT_NAME, ELEVENLABS_MODEL, ELEVENLABS_READY, MAX_CALL_SECONDS, CallConfig, opening_instructions, stt_language
from .metering import book_call
from .accounting import VoiceAccounting
from .callbacks import notify
from .dispatch_guard import DispatchGuard
from runtime import mark_voice_registered, validate_voice_config, voice_enabled

os.environ["LIVEKIT_AGENT_NAME"] = AGENT_NAME

logger = logging.getLogger("voice_call")

DATA_TOPIC = "dentnode.voice"
# Latency fields worth showing, from livekit.agents.llm.chat_context.MetricsReport.
_METRIC_KEYS = (
    "transcription_delay", "end_of_turn_delay", "llm_node_ttft",
    "llm_node_ttfs", "tts_node_ttfb", "e2e_latency",
)

# LiveKit's default keeps one warm process per CPU core (16 on a dev laptop),
# each holding the full import set; that exhausted memory locally. Two warm
# processes cover our call volume; more are spawned on demand.
supervised = voice_enabled()
if supervised:
    validate_voice_config()


def _prewarm_voice_process(_process) -> None:
    if ELEVENLABS_READY:
        # This import otherwise blocks the audio loop on the first Hindi call.
        from livekit.plugins import elevenlabs  # noqa: F401


server = AgentServer(
    num_idle_processes=ServerEnvOption(dev_default=0, prod_default=int(os.environ.get("VOICE_WORKER_IDLE_PROCESSES", "1"))),
    host="127.0.0.1", port=8081,
    setup_fnc=_prewarm_voice_process,
    **({"max_retry": 0, "drain_timeout": 310, "shutdown_process_timeout": 30, "session_end_timeout": 30} if supervised else {}),
)
if supervised:
    server.on("worker_registered", mark_voice_registered)

dispatch_guard = DispatchGuard()


def _reply_llm(model: str) -> lk_openai.LLM:
    extra = {"max_tokens": 180}
    if model not in _REASONING_MANDATORY:
        # Hidden reasoning adds ~0.5 s before the first word on a phone line.
        extra["reasoning"] = {"enabled": False}
    return lk_openai.LLM(
        model=model,
        base_url=settings.openrouter_api_base,
        api_key=settings.openrouter_api_key,
        temperature=0.2,
        extra_body=extra,
        extra_headers={"HTTP-Referer": settings.openrouter_site_url, "X-Title": settings.openrouter_app_name},
    )


@server.rtc_session(on_request=dispatch_guard.on_request)
async def entrypoint(ctx: JobContext) -> None:
    cfg = CallConfig.from_metadata(ctx.job.metadata)
    accounting = VoiceAccounting(cfg, room=ctx.room.name, job_id=ctx.job.id,
                                 dispatch_id=getattr(ctx.job, "dispatch_id", ""))
    try:
        await accounting.checkpoint("worker_assigned")
        await run_call(ctx, cfg, accounting)
    except Exception as error:
        try:
            await accounting.checkpoint("worker_failed", error_type=type(error).__name__,
                monetary_completeness="UNKNOWN", reason="Worker setup failed; absence of usage is not proof of zero expense.")
        except Exception:
            logger.exception("recording worker failure accounting failed")
        await notify(cfg.calling_service_callback, "failed")
        ctx.shutdown(reason="voice_start_failed")
        raise


async def run_call(ctx: JobContext, cfg: CallConfig, accounting: VoiceAccounting) -> None:
    await ctx.connect()
    accepted = False
    transcript: list[str] = []
    transcript_size = 0

    if cfg.tts_model == ELEVENLABS_MODEL:
        from livekit.plugins import elevenlabs
        tts = elevenlabs.TTS(
            voice_id=cfg.tts_voice, model="eleven_flash_v2_5",
            language=cfg.language if cfg.language in ("hi", "en", "mr") else "hi",
        )
    else:
        tts_kwargs = {"voice": cfg.tts_voice} if cfg.tts_voice else {}
        tts = inference.TTS(cfg.tts_model, **tts_kwargs)
    language = stt_language(cfg.stt_model, cfg.language)
    stt = inference.STT(cfg.stt_model, **({"language": language} if language else {}))
    llm = _reply_llm(cfg.reply_model)
    for provider in (stt, llm, tts):
        provider.on("metrics_collected", accounting.capture)
    session = AgentSession(
        stt=stt, llm=llm,
        tts=tts,
    )
    if cfg.calling_service_callback:
        session.input.set_audio_enabled(False)
        session.output.set_audio_enabled(False)

    async def publish(payload: dict) -> None:
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False), reliable=True, topic=DATA_TOPIC)
        except Exception:  # the room may already be closing
            logger.debug("publish failed", exc_info=True)

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        nonlocal transcript_size
        item = event.item
        if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
            return
        text = item.text_content or ""
        if text and transcript_size < 100000:
            line = f"{item.role}: {text}"[:100000 - transcript_size]
            transcript.append(line)
            transcript_size += len(line) + 1
        metrics = {k: round(v * 1000) for k, v in (item.metrics or {}).items()
                   if k in _METRIC_KEYS and isinstance(v, (int, float))}
        asyncio.ensure_future(publish({
            "type": "message", "role": item.role, "text": item.text_content or "",
            "interrupted": bool(getattr(item, "interrupted", False)), "metrics": metrics,
        }))

    @session.on("agent_state_changed")
    def _on_state(event) -> None:
        asyncio.ensure_future(publish({"type": "state", "agent": event.new_state}))

    # Phone-line time: how long each SIP (phone) participant was in the room.
    session_started = time.monotonic()
    sip_joined: dict = {}
    sip_seconds = 0.0
    sip_legs: dict = {}

    def observe_leg(p):
        attributes = p.attributes or {}
        sip_legs[p.sid] = {"participant_sid": p.sid, "sip_call_id": attributes.get("sip.callID"),
                          "provider_sip_call_id": attributes.get("sip.callIDFull"), "trunk_id": attributes.get("sip.trunkID")}

    def _is_sip(p: rtc.RemoteParticipant) -> bool:
        return p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP

    for p in ctx.room.remote_participants.values():
        if _is_sip(p):
            sip_joined[p.identity] = session_started
            observe_leg(p)

    @ctx.room.on("participant_connected")
    def _on_joined(p: rtc.RemoteParticipant) -> None:
        if _is_sip(p):
            sip_joined[p.identity] = time.monotonic()
            observe_leg(p)

    @ctx.room.on("participant_disconnected")
    def _on_left(p: rtc.RemoteParticipant) -> None:
        nonlocal sip_seconds
        joined = sip_joined.pop(p.identity, None)
        if joined is not None:
            sip_seconds += time.monotonic() - joined
            if cfg.calling_service_callback and not sip_joined:
                ctx.shutdown(reason="caller_disconnected")

    async def persist_final_usage() -> None:
        now = time.monotonic()
        try:
            # Flush the provider metrics before taking the immutable snapshot.
            try:
                await asyncio.wait_for(session.aclose(), timeout=5)
            except Exception:
                logger.warning("Voice session close incomplete; retaining observed accounting metrics")
            await accounting.drain()
            await book_call(
                cfg, session.usage, room=ctx.room.name,
                session_seconds=now - session_started,
                sip_seconds=sip_seconds + sum(now - t for t in sip_joined.values()),
                metrics=accounting.metrics, charge_session=accepted,
                accounting={**accounting.identity, "sip_legs": list(sip_legs.values()),
                    "owner_claim_accepted": accepted, "metric_limit_exceeded": accounting.dropped_metrics,
                    "failed_checkpoints": accounting.failed_checkpoints,
                    "ended_at_unix_ms": int(time.time() * 1000)},
            )
            await accounting.checkpoint("terminal", owner_claim_accepted=accepted,
                component_snapshot_persisted=True, metric_count=len(accounting.metrics),
                dropped_metrics=accounting.dropped_metrics, sip_legs=list(sip_legs.values()),
                session_seconds=now - session_started,
                reason="Known usage recorded; invoice, allowances and shared allocations may remain incomplete.")
        except Exception:  # noqa: BLE001 - metering must not fail the job teardown
            logger.exception("booking call usage failed")
            try:
                await accounting.checkpoint("terminal_incomplete", component_snapshot_persisted=False,
                    reason="Final accounting failed; durable request checkpoints remain available.")
            except Exception:
                logger.error("Terminal voice accounting checkpoint failed")

    async def book() -> None:
        # Pinned SDK shutdown/session-end budget is 30 seconds. Keep accounting
        # at 20s plus <=1s incomplete marker and <=7s lifecycle callback. Durable
        # outbox delivery continues in the independent API process after exit.
        try:
            await asyncio.wait_for(persist_final_usage(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("Voice accounting shutdown budget reached; retained checkpoints remain incomplete")
            try:
                await asyncio.wait_for(accounting.checkpoint("terminal_incomplete", component_snapshot_persisted=False,
                    reason="Shutdown accounting budget reached; inspect persisted request evidence."), timeout=1)
            except asyncio.TimeoutError:
                pass
        finally:
            try:
                await asyncio.wait_for(notify(cfg.calling_service_callback, "completed" if accepted else "failed", "\n".join(transcript)), timeout=7)
            except asyncio.TimeoutError:
                logger.warning("Voice lifecycle callback deadline reached; call owner reconciliation remains responsible")

    ctx.add_shutdown_callback(book)

    if cfg.calling_service_callback and not sip_joined:
        ctx.shutdown(reason="caller_disconnected")
        return

    await session.start(
        agent=Agent(instructions=calling_system_prompt(cfg.objective, cfg.recipient_name, language=cfg.language)),
        room=ctx.room,
        # Off regardless of the project setting: LiveKit Cloud observability
        # would upload the call's audio, transcript and logs — patient data —
        # to LiveKit. Recordings belong to Calling Service's own storage.
        record=False,
    )
    logger.info("voice session started")
    accepted = await notify(cfg.calling_service_callback, "started")
    await accounting.checkpoint("owner_claim", accepted=accepted,
        models={"llm": cfg.reply_model, "stt": cfg.stt_model, "tts": cfg.tts_model})
    if not accepted:
        ctx.shutdown(reason="call_claim_rejected")
        return
    # Start billable AI time only once the owner acknowledges the active claim.
    session_started = time.monotonic()
    if cfg.calling_service_callback:
        if not sip_joined:
            ctx.shutdown(reason="caller_disconnected")
            return
        session.input.set_audio_enabled(True)
        session.output.set_audio_enabled(True)
    await publish({"type": "config", "stt": cfg.stt_model, "reply": cfg.reply_model,
                   "tts": cfg.tts_model, "language": cfg.language})

    if cfg.greeting:
        # The disclosure and first greeting must finish before caller speech
        # can interrupt them. Early "hello" otherwise cancels synthesis and
        # can leave the callee hearing silence.
        greeting = session.say(cfg.greeting, allow_interruptions=False)
        logger.info("voice greeting queued")

        def _greeting_done(handle) -> None:
            error = handle.exception()
            logger.info("voice greeting finished", extra={
                "interrupted": handle.interrupted,
                "error_type": type(error).__name__ if error else None,
            })

        greeting.add_done_callback(_greeting_done)
    else:
        session.generate_reply(instructions=opening_instructions(cfg))

    async def hard_stop() -> None:
        await asyncio.sleep(MAX_CALL_SECONDS)
        logger.info("call reached the %ss cap", MAX_CALL_SECONDS)
        ctx.shutdown(reason="max_duration")

    asyncio.ensure_future(hard_stop())
