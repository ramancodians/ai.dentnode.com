"""Centralised configuration for the Laby ADK agent service.

All tunables come from environment variables so the same image runs locally
(docker-compose) and on Cloud Run without code changes.
"""

import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw!r}")


def _openrouter_model(raw: str) -> str:
    """Normalise LABY_MODEL to the litellm `openrouter/...` route.

    Every model goes through OpenRouter, so a bare slug like
    "deepseek/deepseek-v4-flash" (the OpenRouter model id) is prefixed for us.
    This keeps already-deployed env vars working after the cutover.
    """
    slug = raw.strip()
    return slug if slug.startswith("openrouter/") else f"openrouter/{slug}"


class Settings:
    # FastAPI / Cloud Run
    port: int = _int("PORT", 8080)

    # Logging (INFO by default; set DEBUG locally if needed)
    log_level: str = _get("LOG_LEVEL", "INFO")

    # OpenRouter is the ONLY LLM gateway. We never call a provider's API
    # directly — provider credentials live in the OpenRouter dashboard under
    # BYOK, so this service holds exactly one key (OPENROUTER_API_KEY).
    #
    # LABY_MODEL takes an OpenRouter model id ("deepseek/deepseek-v4-flash");
    # the "openrouter/" litellm route prefix is added automatically.
    # IMPORTANT: use a model that supports FUNCTION CALLING. "deepseek-v4-flash"
    # does; "deepseek-reasoner" (R1) does NOT — and the whole agent is
    # built on tool calls, so do not switch to R1.
    model: str = _openrouter_model(_get("LABY_MODEL", "deepseek/deepseek-v4-flash"))

    # Separate model for VISION features (case-from-image, scan review). The
    # default text model is chosen for function calling and cannot see images.
    # Default matches what Node called directly before the migration
    # (gemini-2.5-flash), so extraction quality is unchanged — but routed via
    # OpenRouter, which also removes the Gemini free-tier 20-req/day cap that
    # was causing 500s on the direct path.
    vision_model: str = _openrouter_model(
        _get("LABY_VISION_MODEL", "google/gemini-2.5-flash")
    )

    openrouter_api_key: str = _get("OPENROUTER_API_KEY", "")
    # Override only if pointing at an OpenRouter-compatible proxy.
    openrouter_api_base: str = _get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    # Attribution shown on OpenRouter's activity dashboard / leaderboards.
    openrouter_site_url: str = _get("OPENROUTER_SITE_URL", "https://ai.dentnode.com")
    openrouter_app_name: str = _get("OPENROUTER_APP_NAME", "DentNode Laby")

    # Service-to-service auth. Same secret on both ends:
    #  - Node calls THIS service with x-internal-key.
    #  - THIS service calls Node /internal/laby-tools/* with x-internal-key.
    internal_key: str = _get("INTERNAL_API_KEY", "")

    # Node backend base URL (note the /api mount prefix).
    node_base_url: str = _get("NODE_INTERNAL_BASE_URL", "http://localhost:3000/api")

    # Short-term memory: how many recent thread turns to seed the session with.
    # Clamped to a safe range at read time; validate() checks the raw intent.
    history_turns: int = _int("LABY_HISTORY_TURNS", 12)

    # Maximum seconds a single agent turn may run before being aborted.
    turn_timeout_secs: int = _int("LABY_TURN_TIMEOUT", 120)

    def validate(self) -> None:
        """Raise RuntimeError on any missing or invalid configuration."""
        missing = []
        if not self.internal_key:
            missing.append("INTERNAL_API_KEY")
        if not self.node_base_url:
            missing.append("NODE_INTERNAL_BASE_URL")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        if not (1 <= self.port <= 65535):
            raise RuntimeError(f"PORT must be 1–65535, got {self.port}")
        if not (1 <= self.history_turns <= 100):
            raise RuntimeError(
                f"LABY_HISTORY_TURNS must be 1–100, got {self.history_turns}"
            )
        if self.turn_timeout_secs < 10:
            raise RuntimeError(
                f"LABY_TURN_TIMEOUT must be ≥ 10 seconds, got {self.turn_timeout_secs}"
            )
        if not self.openrouter_api_base.startswith(("http://", "https://")):
            raise RuntimeError(
                "OPENROUTER_API_BASE must be a valid http(s) URL, got: "
                f"{self.openrouter_api_base!r}"
            )


settings = Settings()
