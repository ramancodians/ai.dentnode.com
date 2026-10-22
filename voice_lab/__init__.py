"""Voice Lab — a browser harness for talking to the voice agent and timing it.

GET /voice-lab serves one static page; the page calls POST /voice-lab/turn with
the internal key the developer pastes in. Nothing here stores audio.
"""

from .router import router as voice_lab_router

__all__ = ["voice_lab_router"]
