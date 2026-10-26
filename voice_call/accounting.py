"""PHI-free voice accounting checkpoints in the existing durable usage outbox.

Checkpoints have no money, rating or customer charge. Only the final monetary
component events carry expense, so request evidence cannot double-count cost.
"""
import asyncio
import logging
import math
import time
from types import SimpleNamespace
from typing import Any

import httpx

from agent.config import settings
from agent.d10.usage_outbox import UsageOutbox, build_model_usage_event

logger = logging.getLogger(__name__)
ACCOUNTING_VERSION = "voice-accounting-2026-10-25"
METRIC_FIELDS = (
    "type", "request_id", "timestamp", "duration", "ttft", "ttfb", "cancelled",
    "prompt_tokens", "completion_tokens", "total_tokens", "prompt_cached_tokens",
    "cache_creation_tokens", "reasoning_tokens", "audio_duration", "characters_count",
    "streamed", "connection_reused", "acquire_time", "segment_id", "speech_id",
)


class VoiceAccounting:
    def __init__(self, cfg, *, room: str, job_id: str, dispatch_id: str):
        self.cfg = cfg
        self.identity = {
            "accounting_version": ACCOUNTING_VERSION, "room": room,
            "job_id": job_id, "dispatch_id": dispatch_id,
            "call_id": (cfg.usage_context or {}).get("correlation_id"),
            "source": cfg.source, "language": cfg.language, "voice_id": cfg.tts_voice or None,
            "requested_models": {"llm": cfg.reply_model, "stt": cfg.stt_model, "tts": cfg.tts_model},
            "assigned_at_unix_ms": int(time.time() * 1000),
            **cfg.extra.get("accounting", {}),
        }
        self.metrics: list[dict[str, Any]] = []
        self.tasks: set[asyncio.Task] = set()
        self.sequence = 0
        self.dropped_metrics = 0
        self.failed_checkpoints = 0
        self.outbox = UsageOutbox(settings.d10_usage_outbox_path + ".voice") if cfg.usage_context else None

    async def checkpoint(self, stage: str, **details):
        try:
            await asyncio.wait_for(self._persist_checkpoint(stage, **details), timeout=2)
        except Exception:
            self.failed_checkpoints += 1
            logger.error("Voice accounting checkpoint unavailable; call continues with in-memory usage")

    async def _persist_checkpoint(self, stage: str, **details):
        if not self.outbox:
            return
        self.sequence += 1
        context = SimpleNamespace(**self.cfg.usage_context,
            causation_id=self.cfg.usage_context["source_message_id"], reservation_id=None)
        event = build_model_usage_event(context=context, model_call_index=self.sequence,
            attempt=1, provider="livekit", model="voice-accounting", provider_request_id=None,
            usage=None, raw_usage={**self.identity, "stage": stage, **details},
            latency_ms=0, status="recorded", feature="voice_call")
        event.update(event_type="ai.voice_accounting", gateway="livekit",
            idempotency_key=f"voice-accounting:{context.correlation_id}:{self.identity['job_id']}:{self.sequence}")
        event.pop("pricing", None)
        await self.outbox.initialize()
        await self.outbox.enqueue(event)

    def capture(self, metric):
        # SDK metric models contain usage only. An explicit allowlist prevents
        # future plugin fields from uploading text, audio, headers or secrets.
        row = {key: getattr(metric, key) for key in METRIC_FIELDS if hasattr(metric, key)}
        row = {key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in row.items()}
        metadata = getattr(metric, "metadata", None)
        if metadata:
            row["model"] = getattr(metadata, "model_name", None)
            row["provider"] = getattr(metadata, "model_provider", None)
        if row.get("type") not in ("llm_metrics", "stt_metrics", "tts_metrics"):
            return
        if len(self.metrics) >= 512:
            self.dropped_metrics += 1
            return
        self.metrics.append(row)
        task = asyncio.create_task(self.checkpoint("provider_metric", metric=row))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(self._finished)

    @staticmethod
    def _finished(task):
        if not task.cancelled() and task.exception():
            logger.error("Voice accounting checkpoint could not be persisted")

    async def drain(self):
        if self.tasks:
            await asyncio.wait(list(self.tasks), timeout=2)


async def generation_receipts(metrics: list[dict]) -> dict[str, dict]:
    """Read-only, bounded receipts lookup. Failure never discards original usage."""
    ids = list(dict.fromkeys(m.get("request_id") for m in metrics
        if m.get("type") == "llm_metrics" and isinstance(m.get("request_id"), str)
        and m["request_id"].startswith("gen-") and len(m["request_id"]) <= 256))[:64]
    receipts: dict[str, dict] = {}
    semaphore = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=3, follow_redirects=False) as client:
        async def read(request_id):
            async with semaphore:
                try:
                    response = await client.get(f"{settings.openrouter_api_base}/generation",
                        params={"id": request_id}, headers={"Authorization": f"Bearer {settings.openrouter_api_key}"})
                    if response.status_code != 200:
                        return
                    data = response.json().get("data") or {}
                    cost = data.get("total_cost")
                    if data.get("id") != request_id or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                        return
                    receipts[request_id] = {key: data.get(key) for key in (
                        "id", "model", "provider_name", "total_cost", "tokens_prompt", "tokens_completion",
                        "native_tokens_prompt", "native_tokens_completion", "native_tokens_reasoning",
                        "native_tokens_cached", "generation_time", "latency", "created_at", "finish_reason")}
                except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                    return
        tasks = [asyncio.create_task(read(request_id)) for request_id in ids]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=6)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    return receipts
