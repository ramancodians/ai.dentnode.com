"""SQLite-backed, retrying outbox for authoritative D10 AI usage events."""

import asyncio
import json
import logging
import sqlite3
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

import httpx

from agent.config import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS d10_usage_outbox (
    event_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at REAL NOT NULL,
    created_at REAL NOT NULL,
    delivered_at REAL,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS d10_usage_outbox_pending
    ON d10_usage_outbox(delivered_at, available_at, created_at);
"""


def _money_fields(cost: Optional[float]) -> tuple[Optional[str], Optional[int]]:
    if cost is None:
        return None, None
    try:
        value = Decimal(str(cost))
    except (InvalidOperation, ValueError):
        return None, None
    return format(value, "f"), int(
        (value * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def build_model_usage_event(
    *,
    context: Any,
    model_call_index: int,
    attempt: int,
    provider: str,
    model: str,
    provider_request_id: Optional[str],
    usage: Optional[Dict[str, Any]],
    raw_usage: Optional[Dict[str, Any]],
    latency_ms: int,
    status: str,
    cost_usd: Optional[float] = None,
    error_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a PHI-free event. D10, not this service, applies its rate card."""
    usage = usage or {}
    cost_text, cost_micros = _money_fields(cost_usd)
    provider_id = provider_request_id or "unassigned"
    idempotency_key = (
        f"d10-agent:{context.correlation_id}:{model_call_index}:{attempt}:{provider_id}"
    )
    now = time.time()
    return {
        "event_id": str(uuid4()),
        "idempotency_key": idempotency_key,
        "event_type": "ai.model_call",
        "occurred_at_unix_ms": int(now * 1000),
        "clinic_id": context.clinic_id,
        "user_id": context.user_id,
        "actor_id": context.actor_id,
        "conversation_id": context.conversation_id,
        "source_message_id": context.source_message_id,
        "correlation_id": context.correlation_id,
        "causation_id": context.causation_id,
        "reservation_id": context.reservation_id,
        "provider": provider,
        "gateway": "openrouter",
        "model": model,
        "provider_request_id": provider_request_id,
        "model_call_index": model_call_index,
        "attempt": attempt,
        "status": status,
        "latency_ms": latency_ms,
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "modal_units": {
            "image_input": int(usage.get("image_input_units") or 0),
            "image_output": int(usage.get("image_output_units") or 0),
            "audio_input": int(usage.get("audio_input_units") or 0),
            "audio_output": int(usage.get("audio_output_units") or 0),
            "video_input": int(usage.get("video_input_units") or 0),
            "video_output": int(usage.get("video_output_units") or 0),
        },
        "provider_cost_usd": cost_text,
        "provider_cost_micros_usd": cost_micros,
        "raw_usage": raw_usage or {},
        "error_code": error_code,
    }


class UsageOutbox:
    """Persist first, then deliver batches with at-least-once semantics."""

    def __init__(
        self,
        path: str,
        *,
        endpoint: Optional[str] = None,
        internal_key: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.path = path
        self.endpoint = endpoint or (
            f"{settings.d10_internal_base_url}/internal/ai/usage-events"
        )
        self.internal_key = internal_key or settings.d10_internal_key
        self.batch_size = batch_size or settings.d10_usage_batch_size
        self._flush_lock = asyncio.Lock()
        self._stop = asyncio.Event()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    async def initialize(self) -> None:
        def _initialize() -> None:
            parent = Path(self.path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)

        await asyncio.to_thread(_initialize)

    async def enqueue(self, event: Dict[str, Any]) -> None:
        payload = json.dumps(event, separators=(",", ":"), sort_keys=True)

        def _enqueue() -> None:
            now = time.time()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO d10_usage_outbox
                      (event_id, idempotency_key, payload, available_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event["event_id"],
                        event["idempotency_key"],
                        payload,
                        now,
                        now,
                    ),
                )

        await asyncio.to_thread(_enqueue)

    async def _pending(self) -> list[sqlite3.Row]:
        def _read() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return list(
                    connection.execute(
                        """
                        SELECT event_id, payload, attempts
                        FROM d10_usage_outbox
                        WHERE delivered_at IS NULL AND available_at <= ?
                        ORDER BY created_at ASC
                        LIMIT ?
                        """,
                        (time.time(), self.batch_size),
                    )
                )

        return await asyncio.to_thread(_read)

    async def _mark_delivered(self, event_ids: Iterable[str]) -> None:
        ids = list(event_ids)
        if not ids:
            return

        def _mark() -> None:
            with self._connect() as connection:
                connection.executemany(
                    "UPDATE d10_usage_outbox SET delivered_at = ?, last_error = NULL WHERE event_id = ?",
                    [(time.time(), event_id) for event_id in ids],
                )

        await asyncio.to_thread(_mark)

    async def _mark_retry(self, rows: Iterable[sqlite3.Row], error: str) -> None:
        values = []
        now = time.time()
        for row in rows:
            attempts = int(row["attempts"]) + 1
            delay = min(300.0, float(2 ** min(attempts, 8)))
            values.append((attempts, now + delay, error[:1000], row["event_id"]))

        def _mark() -> None:
            with self._connect() as connection:
                connection.executemany(
                    """
                    UPDATE d10_usage_outbox
                    SET attempts = ?, available_at = ?, last_error = ?
                    WHERE event_id = ?
                    """,
                    values,
                )

        await asyncio.to_thread(_mark)

    async def flush_once(self) -> int:
        async with self._flush_lock:
            rows = await self._pending()
            if not rows:
                return 0
            events = [json.loads(row["payload"]) for row in rows]
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(
                        self.endpoint,
                        json={"events": events},
                        headers={
                            "x-d10-internal-key": self.internal_key,
                            "Content-Type": "application/json",
                        },
                    )
                if response.status_code >= 400:
                    raise RuntimeError(f"D10 usage endpoint returned {response.status_code}")
            except Exception as exc:  # noqa: BLE001 - retain and retry every event
                await self._mark_retry(rows, str(exc))
                logger.warning(
                    "D10 usage outbox delivery failed",
                    extra={"events": len(rows), "error": str(exc)},
                )
                return 0

            await self._mark_delivered(row["event_id"] for row in rows)
            return len(rows)

    async def run(self, interval_secs: int) -> None:
        self._stop.clear()
        while not self._stop.is_set():
            await self.flush_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval_secs)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._stop.set()

    async def pending_count(self) -> int:
        def _count() -> int:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM d10_usage_outbox WHERE delivered_at IS NULL"
                ).fetchone()
                return int(row["count"])

        return await asyncio.to_thread(_count)
