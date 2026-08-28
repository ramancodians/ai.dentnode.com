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

    # Separate model for VISION features (case-from-image, rejected-cases). The
    # default text model is chosen for function calling and cannot see images.
    # Default matches what Node called directly before the migration
    # (gemini-2.5-flash), so extraction quality is unchanged — but routed via
    # OpenRouter, which also removes the Gemini free-tier 20-req/day cap that
    # was causing 500s on the direct path.
    vision_model: str = _openrouter_model(
        _get("LABY_VISION_MODEL", "google/gemini-2.5-flash")
    )

    # Vision model for the SCAN REVIEW specifically. Deliberately its own knob:
    # scan review is evaluated and tuned against dental arch renders, and its
    # model should be swappable without touching case-from-image or
    # rejected-cases, which share `vision_model` and are tuned for a different
    # job. Falls back to LABY_VISION_MODEL so an unset deploy keeps working.
    scan_review_vision_model: str = _openrouter_model(
        _get("SCAN_REVIEW_VISION_MODEL", "")
        or _get("LABY_VISION_MODEL", "google/gemini-2.5-flash")
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

    # D10.live is a separate trusted caller/ tool host.  Keep these settings
    # distinct from the app.dentnode.com (Laby) integration above so either
    # service can rotate credentials or move independently.  The key falls
    # back to INTERNAL_API_KEY during the migration, but production should set
    # D10_INTERNAL_KEY explicitly.
    d10_internal_key: str = _get("D10_INTERNAL_KEY", "") or internal_key
    d10_internal_base_url: str = _get(
        "D10_INTERNAL_BASE_URL", "http://localhost:3000/api"
    ).rstrip("/")
    d10_agent_model: str = _openrouter_model(
        _get("D10_AGENT_MODEL", "")
        or _get("LABY_MODEL", "deepseek/deepseek-v4-flash")
    )
    d10_agent_timeout_secs: int = _int("D10_AGENT_TIMEOUT_SECS", 120)
    d10_agent_max_model_calls: int = _int("D10_AGENT_MAX_MODEL_CALLS", 8)

    # SQLite is an intentionally small durable outbox, not the billing source
    # of truth.  In Cloud Run this path should point at a mounted persistent
    # volume; local development may use the repository-local default.
    d10_usage_outbox_path: str = _get(
        "D10_USAGE_OUTBOX_PATH", ".data/d10-usage-outbox.sqlite3"
    )
    d10_usage_flush_interval_secs: int = _int(
        "D10_USAGE_FLUSH_INTERVAL_SECS", 10
    )
    d10_usage_batch_size: int = _int("D10_USAGE_BATCH_SIZE", 100)

    # Audio-to-Text agent: transcribe a Digital Ocean audio URL and summarise
    # it. A single-call feature agent (no ADK tool loop) shared by both
    # app.dentnode.com and d10.live. The audio model must accept multimodal
    # `input_audio` content parts — the OpenRouter catalog currently exposes
    # `openai/gpt-audio` / `openai/gpt-audio-mini` for this.
    audio_to_text_model: str = _openrouter_model(
        _get("AUDIO_TO_TEXT_MODEL", "openai/gpt-audio-mini")
    )
    audio_to_text_timeout_secs: int = _int("AUDIO_TO_TEXT_TIMEOUT_SECS", 120)
    audio_to_text_max_bytes: int = _int("AUDIO_TO_TEXT_MAX_BYTES", 25 * 1024 * 1024)
    audio_to_text_fetch_timeout_secs: int = _int(
        "AUDIO_TO_TEXT_FETCH_TIMEOUT_SECS", 30
    )
    # Comma-separated host allowlist for audio downloads. Defaults to Digital
    # Ocean Spaces (origin + CDN). A leading-dot entry matches any subdomain.
    audio_to_text_allowed_hosts: tuple = tuple(
        h.strip().lower()
        for h in _get(
            "AUDIO_TO_TEXT_ALLOWED_HOSTS",
            ".digitaloceanspaces.com,.cdn.digitaloceanspaces.com",
        ).split(",")
        if h.strip()
    )
    # Local-dev escape hatch (mirrors SCAN_REVIEW_ALLOW_INSECURE_FETCH): permits
    # http:// and non-public hosts so a local sample file can be transcribed.
    audio_to_text_allow_insecure_fetch: bool = (
        _get("AUDIO_TO_TEXT_ALLOW_INSECURE_FETCH", "0").lower() in ("1", "true", "yes")
    )

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
        if not self.d10_internal_key:
            raise RuntimeError(
                "D10_INTERNAL_KEY (or fallback INTERNAL_API_KEY) is required"
            )
        if not self.d10_internal_base_url.startswith(("http://", "https://")):
            raise RuntimeError(
                "D10_INTERNAL_BASE_URL must be a valid http(s) URL, got: "
                f"{self.d10_internal_base_url!r}"
            )
        if self.d10_agent_timeout_secs < 10:
            raise RuntimeError(
                "D10_AGENT_TIMEOUT_SECS must be ≥ 10 seconds, got "
                f"{self.d10_agent_timeout_secs}"
            )
        if not (1 <= self.d10_agent_max_model_calls <= 32):
            raise RuntimeError(
                "D10_AGENT_MAX_MODEL_CALLS must be 1–32, got "
                f"{self.d10_agent_max_model_calls}"
            )
        if self.d10_usage_flush_interval_secs < 1:
            raise RuntimeError("D10_USAGE_FLUSH_INTERVAL_SECS must be at least 1")
        if not (1 <= self.d10_usage_batch_size <= 1000):
            raise RuntimeError("D10_USAGE_BATCH_SIZE must be 1–1000")
        if self.audio_to_text_timeout_secs < 10:
            raise RuntimeError(
                "AUDIO_TO_TEXT_TIMEOUT_SECS must be ≥ 10 seconds, got "
                f"{self.audio_to_text_timeout_secs}"
            )
        if not (1 <= self.audio_to_text_max_bytes <= 100 * 1024 * 1024):
            raise RuntimeError(
                "AUDIO_TO_TEXT_MAX_BYTES must be 1–100 MiB, got "
                f"{self.audio_to_text_max_bytes}"
            )
        if self.audio_to_text_fetch_timeout_secs < 1:
            raise RuntimeError(
                "AUDIO_TO_TEXT_FETCH_TIMEOUT_SECS must be at least 1"
            )
        if not self.openrouter_api_base.startswith(("http://", "https://")):
            raise RuntimeError(
                "OPENROUTER_API_BASE must be a valid http(s) URL, got: "
                f"{self.openrouter_api_base!r}"
            )


settings = Settings()
