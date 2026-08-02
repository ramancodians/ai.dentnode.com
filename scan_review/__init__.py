"""Scan Review — standalone 3D scan QA for DentNode.

Takes raw mesh file URLs (STL and friends), downloads them safely, measures the
geometry, and returns an actionable QA review: holes, non-manifold edges,
floating fragments, winding, scale sanity, and a reproducible 0–100 score.

This is **not** part of Laby. It has no agent, no ADK session, and no tools — it
is a plain request/response pipeline that happens to live in the same service.
It borrows only two pieces of shared plumbing (the OpenRouter gateway and the
usage ledger client) because duplicating either would be worse than the coupling.

Layout:
    config.py    — SCAN_REVIEW_* settings, independent of the agent's
    fetch.py     — SSRF-guarded downloader (the security boundary)
    geometry.py  — deterministic mesh measurement + findings + score
    review.py    — orchestration; geometry decides, the model describes
    router.py    — FastAPI routes under /scan-review/*

Mount with:
    from scan_review import scan_review_router
    app.include_router(scan_review_router)
"""

from .config import settings
from .fetch import MeshFetchError, fetch_mesh
from .geometry import (
    MeshInspectError,
    MeshReport,
    build_findings,
    inspect_mesh,
    score_report,
)
from .review import ScanReviewError, ScanReviewResult, generate_scan_review

# Exported under a qualified name on purpose. `from .router import router` would
# bind the APIRouter object to `scan_review.router`, shadowing the submodule of
# the same name — after which `import scan_review.router` hands back the router
# instead of the module, and patching anything inside it fails.
from .router import router as scan_review_router

__all__ = [
    "MeshFetchError",
    "MeshInspectError",
    "MeshReport",
    "ScanReviewError",
    "ScanReviewResult",
    "build_findings",
    "fetch_mesh",
    "generate_scan_review",
    "inspect_mesh",
    "scan_review_router",
    "score_report",
    "settings",
]
