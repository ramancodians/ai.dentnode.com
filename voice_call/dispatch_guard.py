"""Keep one speaking agent for each LiveKit room dispatch."""

import logging
import time

logger = logging.getLogger("voice_call")


class DispatchGuard:
    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], float] = {}

    async def on_request(self, request) -> None:
        now = time.monotonic()
        # Rooms are unique per call. Keep completed rooms long enough to reject
        # delayed duplicate offers without retaining them indefinitely.
        self._seen = {key: expiry for key, expiry in self._seen.items() if expiry > now}
        key = (request.room.sid or request.room.name, request.job.dispatch_id or request.room.name)
        if key in self._seen:
            logger.warning("rejecting duplicate voice job", extra={"room": request.room.name,
                         "dispatch_id": request.job.dispatch_id, "job_id": request.id})
            await request.reject(terminate=True)
            return

        self._seen[key] = now + 3600
        try:
            # Also prevent two workers from publishing as separate agents.
            await request.accept(identity=f"dentnode-voice-{request.room.sid or request.room.name}")
        except Exception:
            self._seen.pop(key, None)
            raise
