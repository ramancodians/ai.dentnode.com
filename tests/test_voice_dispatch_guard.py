import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voice_call.dispatch_guard import DispatchGuard


def request(job_id: str, dispatch_id: str = "dispatch-1"):
    return SimpleNamespace(
        id=job_id,
        room=SimpleNamespace(sid="room-1", name="calling-one"),
        job=SimpleNamespace(dispatch_id=dispatch_id),
        accept=AsyncMock(), reject=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_two_jobs_for_one_dispatch_only_accept_one():
    guard = DispatchGuard()
    first = request("job-1")
    duplicate = request("job-2")

    await asyncio.gather(guard.on_request(first), guard.on_request(duplicate))

    first.accept.assert_awaited_once_with(identity="dentnode-voice-room-1")
    duplicate.accept.assert_not_awaited()
    duplicate.reject.assert_awaited_once_with(terminate=True)


@pytest.mark.asyncio
async def test_failed_accept_does_not_reserve_room():
    guard = DispatchGuard()
    first = request("job-1")
    first.accept.side_effect = RuntimeError("assignment failed")
    with pytest.raises(RuntimeError, match="assignment failed"):
        await guard.on_request(first)

    replacement = request("job-2")
    await guard.on_request(replacement)
    replacement.accept.assert_awaited_once()
