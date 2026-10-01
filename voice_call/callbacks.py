"""Bounded, server-origin-only lifecycle callbacks for the shared call owner."""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallCallback:
    url: str
    token: str = field(repr=False)

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Invalid call lifecycle callback")
        origin = urlsplit(os.environ.get("CALLING_SERVICE_URL", ""))
        url, token = value.get("url"), value.get("token")
        if not isinstance(url, str) or not isinstance(token, str):
            raise ValueError("Invalid call lifecycle callback")
        target = urlsplit(url)
        if (origin.scheme not in ("http", "https") or not origin.hostname
                or target.scheme != origin.scheme or target.netloc != origin.netloc
                or target.username or target.password or target.query or target.fragment
                or not re.fullmatch(r"/internal/ai-answering/[A-Za-z0-9_-]+", target.path)
                or not re.fullmatch(r"[a-fA-F0-9]{64}", token)):
            raise ValueError("Call lifecycle callback must use the configured Calling Service origin")
        return cls(url, token)


async def notify(callback, status: str, transcript: str = "") -> bool:
    """False means the owner rejected/failed to acknowledge the claim; never speak then."""
    if callback is None:
        return True
    body = {"status": status}
    if transcript and status != "started":
        body["transcript"] = transcript[:100000]
    async with httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False) as client:
        for attempt in range(3):
            try:
                response = await client.post(callback.url, headers={"Authorization": f"Bearer {callback.token}"}, json=body)
                if response.status_code == 409:
                    return False
                if response.is_success:
                    return True
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    break
            except httpx.HTTPError:
                pass
            if attempt < 2:
                await asyncio.sleep(0.25 * (attempt + 1))
    # Do not include URLs, tokens, request bodies or provider exception strings.
    logger.error("Call lifecycle acknowledgement failed: status=%s", status)
    return False
