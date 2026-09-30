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

from .config import AGENT_NAME, MAX_CALL_SECONDS, CallConfig, opening_instructions, stt_language
from .metering import book_call

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
server = AgentServer(num_idle_processes=ServerEnvOption(dev_default=0, prod_default=2))


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


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    cfg = CallConfig.from_metadata(ctx.job.metadata)
    await ctx.connect()

    tts_kwargs = {"voice": cfg.tts_voice} if cfg.tts_voice else {}
    language = stt_language(cfg.stt_model, cfg.language)
    session = AgentSession(
        stt=inference.STT(cfg.stt_model, **({"language": language} if language else {})),
        llm=_reply_llm(cfg.reply_model),
        tts=inference.TTS(cfg.tts_model, **tts_kwargs),
    )

    async def publish(payload: dict) -> None:
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False), reliable=True, topic=DATA_TOPIC)
        except Exception:  # the room may already be closing
            logger.debug("publish failed", exc_info=True)

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        item = event.item
        if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
            return
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

    def _is_sip(p: rtc.RemoteParticipant) -> bool:
        return p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP

    for p in ctx.room.remote_participants.values():
        if _is_sip(p):
            sip_joined[p.identity] = session_started

    @ctx.room.on("participant_connected")
    def _on_joined(p: rtc.RemoteParticipant) -> None:
        if _is_sip(p):
            sip_joined[p.identity] = time.monotonic()

    @ctx.room.on("participant_disconnected")
    def _on_left(p: rtc.RemoteParticipant) -> None:
        nonlocal sip_seconds
        joined = sip_joined.pop(p.identity, None)
        if joined is not None:
            sip_seconds += time.monotonic() - joined

    async def book() -> None:
        now = time.monotonic()
        try:
            await book_call(
                cfg, session.usage, room=ctx.room.name,
                session_seconds=now - session_started,
                sip_seconds=sip_seconds + sum(now - t for t in sip_joined.values()),
            )
        except Exception:  # noqa: BLE001 - metering must not fail the job teardown
            logger.exception("booking call usage failed")

    ctx.add_shutdown_callback(book)

    await session.start(
        agent=Agent(instructions=calling_system_prompt(cfg.objective, cfg.recipient_name, multilingual=True)),
        room=ctx.room,
        # Off regardless of the project setting: LiveKit Cloud observability
        # would upload the call's audio, transcript and logs — patient data —
        # to LiveKit. Recordings belong to Calling Service's own storage.
        record=False,
    )
    await publish({"type": "config", "stt": cfg.stt_model, "reply": cfg.reply_model,
                   "tts": cfg.tts_model, "language": cfg.language})

    if cfg.greeting:
        session.say(cfg.greeting)
    else:
        session.generate_reply(instructions=opening_instructions(cfg))

    async def hard_stop() -> None:
        await asyncio.sleep(MAX_CALL_SECONDS)
        logger.info("call reached the %ss cap", MAX_CALL_SECONDS)
        ctx.shutdown(reason="max_duration")

    asyncio.ensure_future(hard_stop())
