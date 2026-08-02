"""Scan QA — segmentation-conditioned dental scan quality assessment.

Hybrid by design: a 3D model segments the mesh into teeth and gingiva, then
deterministic geometry decides whether the scan is acceptable. No neural network
is asked to output "good scan" / "bad scan" directly.

Backbone is pluggable via the `Segmenter` protocol — MeshSegNet today, Point
Transformer V3 once DentNode has enough labelled scans to fine-tune on.
"""

from .checks import Finding
from .pipeline import FileQA, ScanQAReport, ScanRole, run_scan_qa
from .router import router as scan_qa_router
from .segmentation import (
    MeshSegNetSegmenter,
    Segmentation,
    Segmenter,
    ToothRegion,
)

__all__ = [
    "Finding",
    "FileQA",
    "ScanQAReport",
    "ScanRole",
    "run_scan_qa",
    "scan_qa_router",
    "MeshSegNetSegmenter",
    "Segmentation",
    "Segmenter",
    "ToothRegion",
]
