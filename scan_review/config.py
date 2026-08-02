"""Configuration for the standalone Scan Review module.

Deliberately separate from `agent/config.py`. Scan Review is not part of Laby
and does not share its tunables — it has its own model choice, its own limits,
and its own security posture (it is the only component in this service that
fetches a caller-supplied URL). Keeping the knobs here means a change to scan
review can never accidentally alter the agent's behaviour.

Every variable is prefixed `SCAN_REVIEW_` so the two namespaces stay obvious in
Cloud Run's env list. The two things it does inherit are the OpenRouter
credentials and the internal service key, because those are properties of the
*deployment*, not of either module.
"""

import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got: {raw!r}"
        )


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ScanReviewSettings:
    """Settings are read at import time, matching the rest of the service."""

    # ── Model ─────────────────────────────────────────────────────────────
    # The mesh report is text/JSON, not images, so this does NOT need a vision
    # model — which makes it far cheaper than the Laby vision scan review.
    # Reads LABY_MODEL as a fallback only so a fresh deploy still works before
    # the dedicated variable is set.
    model: str = _get("SCAN_REVIEW_MODEL", "") or _get(
        "LABY_MODEL", "deepseek/deepseek-v4-flash"
    )

    # Whether the LLM narrative layer runs at all. Geometry always runs; with
    # this off the endpoint returns measurements and deterministic findings
    # only, costs nothing, and never calls OpenRouter. Useful for load tests
    # and for labs on a plan without AI.
    enable_llm: bool = _bool("SCAN_REVIEW_ENABLE_LLM", True)

    llm_timeout_secs: int = _int("SCAN_REVIEW_LLM_TIMEOUT", 90)

    # ── Fetch limits (security controls — see fetch.py) ───────────────────
    # Hard cap on one downloaded mesh. Dental STLs run 5–40 MB; well past that
    # is either not a scan or a denial-of-service attempt.
    max_bytes: int = _int("SCAN_REVIEW_MAX_BYTES", 80 * 1024 * 1024)

    # Meshes per request. A full case is usually upper + lower (+ bite).
    max_files: int = _int("SCAN_REVIEW_MAX_FILES", 8)

    # Per-file download timeout, seconds.
    fetch_timeout_secs: int = _int("SCAN_REVIEW_FETCH_TIMEOUT", 60)

    # Wall-clock cap for analysing ONE mesh. Geometry work is CPU-bound and runs
    # in a worker thread; a pathological mesh must not pin that thread forever.
    analyze_timeout_secs: int = _int("SCAN_REVIEW_ANALYZE_TIMEOUT", 120)

    # Refuse absurd triangle counts before trimesh allocates for them. 5M faces
    # is already ~10x a normal full-arch intraoral scan.
    max_faces: int = _int("SCAN_REVIEW_MAX_FACES", 5_000_000)

    # Comma-separated host allowlist (the S3/CDN buckets that hold scans).
    # Empty = any public host that passes the SSRF checks. Setting this in
    # production is strongly recommended: it turns the guard from "block the
    # obviously internal" into "only reach our own storage".
    allowed_hosts_raw: str = _get("SCAN_REVIEW_ALLOWED_HOSTS", "")

    # LOCAL DEV ONLY. Permits http:// and private/loopback targets so a mesh can
    # be served from a dev box. Must never be enabled in a deployed environment.
    allow_insecure_fetch: bool = _bool("SCAN_REVIEW_ALLOW_INSECURE_FETCH", False)

    @property
    def allowed_hosts(self) -> frozenset:
        """Parsed allowlist. An empty set means "no allowlist configured"."""
        return frozenset(
            h.strip().lower()
            for h in self.allowed_hosts_raw.split(",")
            if h.strip()
        )

    def validate(self) -> None:
        """Raise RuntimeError on nonsensical configuration."""
        if self.max_bytes < 1024:
            raise RuntimeError(
                f"SCAN_REVIEW_MAX_BYTES must be at least 1024, got {self.max_bytes}"
            )
        if not (1 <= self.max_files <= 64):
            raise RuntimeError(
                f"SCAN_REVIEW_MAX_FILES must be 1–64, got {self.max_files}"
            )
        if self.fetch_timeout_secs < 1:
            raise RuntimeError(
                "SCAN_REVIEW_FETCH_TIMEOUT must be ≥ 1 second, got "
                f"{self.fetch_timeout_secs}"
            )
        if self.analyze_timeout_secs < 1:
            raise RuntimeError(
                "SCAN_REVIEW_ANALYZE_TIMEOUT must be ≥ 1 second, got "
                f"{self.analyze_timeout_secs}"
            )
        if self.max_faces < 4:
            raise RuntimeError(
                f"SCAN_REVIEW_MAX_FACES must be ≥ 4, got {self.max_faces}"
            )


settings = ScanReviewSettings()
