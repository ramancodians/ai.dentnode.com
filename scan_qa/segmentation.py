"""Segmentation backends for Scan QA.

The pipeline depends on the `Segmenter` protocol, never on a concrete model, so
the backbone can be swapped (MeshSegNet now, Point Transformer V3 later) without
touching the QC rules that consume the labels.

Contract: a segmenter takes a trimesh mesh and returns a `Segmentation` whose
`labels` array is per-face on the *returned decimated mesh* — not on the input
mesh. Segmentation runs on a reduced mesh (MeshSegNet is trained at 10k cells);
the QC rules that need full resolution work off the original mesh separately and
use the segmentation only for region attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# MeshSegNet class 0 is gingiva; 1..14 are teeth, excluding third molars.
GINGIVA = 0
NUM_TOOTH_CLASSES = 14

# FDI numbering along the arch, patient's right to patient's left.
FDI_UPPER = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
FDI_LOWER = [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37]


@dataclass
class ToothRegion:
    """One segmented tooth on the decimated mesh."""

    label: int
    face_indices: np.ndarray = field(repr=False)
    centroid: np.ndarray = field(repr=False)
    area_mm2: float
    # Position along the arch, 0 = patient's right-most. Assigned by ordering
    # regions around the arch, which is more reliable than the raw class id.
    arch_index: Optional[int] = None
    fdi: Optional[int] = None
    # Mean softmax confidence over the region's faces.
    confidence: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "label": int(self.label),
            "arch_index": self.arch_index,
            "fdi": self.fdi,
            "faces": int(len(self.face_indices)),
            "area_mm2": round(self.area_mm2, 2),
            "confidence": round(self.confidence, 3),
            "centroid": [round(float(c), 2) for c in self.centroid],
        }


@dataclass
class Segmentation:
    mesh: trimesh.Trimesh = field(repr=False)          # decimated mesh
    labels: np.ndarray = field(repr=False)             # per-face, len == len(mesh.faces)
    probs: Optional[np.ndarray] = field(default=None, repr=False)
    jaw: str = "upper"
    backend: str = "unknown"
    teeth: List[ToothRegion] = field(default_factory=list)

    @property
    def gingiva_fraction(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float((self.labels == GINGIVA).mean())

    @property
    def mean_confidence(self) -> float:
        if self.probs is None or len(self.probs) == 0:
            return 0.0
        return float(self.probs.max(axis=-1).mean())

    def as_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "jaw": self.jaw,
            "cells": int(len(self.labels)),
            "teeth_detected": len(self.teeth),
            "gingiva_fraction": round(self.gingiva_fraction, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "teeth": [t.as_dict() for t in self.teeth],
        }


class Segmenter(Protocol):
    name: str

    def segment(self, mesh: trimesh.Trimesh, jaw: str) -> Segmentation: ...


# ── MeshSegNet ────────────────────────────────────────────────────────────────

TARGET_CELLS = 10_000
_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "meshsegnet"


class MeshSegNetSegmenter:
    """MeshSegNet (Lian et al. 2020), MIT-licensed, vendored under vendor/.

    Runs on CPU in roughly 10s per arch at 10k cells, so it needs no GPU to
    prototype with.

    KNOWN DOMAIN GAP — read before trusting per-tooth output on a prep case:
    the published weights are trained on 72 scans of *natural* dentition. A
    prepared tooth has had its anatomy cut away and does not look like a tooth
    to this model; in DentNode testing the prepared quadrant was largely
    labelled gingiva while the opposing natural arch segmented cleanly. Treat
    tooth labels on a prepared arch as unreliable until the model is fine-tuned
    on DentNode's own labelled preps.
    """

    name = "meshsegnet"

    def __init__(self, vendor_dir: Path = _VENDOR, device: str = "cpu"):
        self.vendor_dir = Path(vendor_dir)
        self.device = device
        self._models: Dict[str, object] = {}

    # -- model loading -------------------------------------------------------

    def _load(self, jaw: str):
        """Lazy-load per jaw. 'Max' = maxillary/upper, 'Man' = mandibular/lower."""
        key = "Max" if jaw == "upper" else "Man"
        if key in self._models:
            return self._models[key]

        import sys

        import torch

        if str(self.vendor_dir) not in sys.path:
            sys.path.insert(0, str(self.vendor_dir))
        from meshsegnet import MeshSegNet  # type: ignore

        weights = self.vendor_dir / f"MeshSegNet_{key}.zip"
        if not weights.exists():
            raise FileNotFoundError(
                f"MeshSegNet weights missing: {weights}. Fetch them from "
                "github.com/Tai-Hsien/MeshSegNet (models/)."
            )

        model = MeshSegNet(num_classes=15, num_channels=15)
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self._models[key] = model
        logger.info("MeshSegNet loaded", extra={"jaw": jaw, "epoch": ck.get("epoch")})
        return model

    # -- preprocessing -------------------------------------------------------

    @staticmethod
    def _features(mesh: trimesh.Trimesh):
        """15 channels: 3 vertex coords x3, barycenter, normal — per step5_predict.py.

        The normalisation is reproduced exactly; changing it silently degrades
        accuracy because the weights were trained against these statistics.
        """
        points = np.asarray(mesh.vertices, dtype=np.float64).copy()
        faces = np.asarray(mesh.faces)
        points -= points.mean(axis=0)

        cells = points[faces].reshape(len(faces), 9).astype("float32")
        normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
        barycenters = points[faces].mean(axis=1)

        maxs, mins = points.max(axis=0), points.min(axis=0)
        means, stds = points.mean(axis=0), points.std(axis=0)
        nmeans, nstds = normals.mean(axis=0), normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        return np.column_stack((cells, barycenters, normals)).astype("float32")

    @staticmethod
    def _adjacency(X: np.ndarray):
        """Row-normalised neighbourhood matrices at 0.1 and 0.2 (normalised units).

        Held in float32: at 10k cells each matrix is 400 MB in float64 and the
        pair will not fit comfortably alongside the model on a small box.
        """
        bc = X[:, 9:12]
        d = np.linalg.norm(bc[:, None, :] - bc[None, :, :], axis=-1).astype("float32")
        mats = []
        for thresh in (0.1, 0.2):
            A = (d < thresh).astype("float32")
            A /= np.maximum(A.sum(axis=1, keepdims=True), 1.0)
            mats.append(A)
        del d
        return mats

    # -- inference -----------------------------------------------------------

    def segment(self, mesh: trimesh.Trimesh, jaw: str = "upper") -> Segmentation:
        import torch

        model = self._load(jaw)

        work = mesh
        if len(work.faces) > TARGET_CELLS:
            work = work.simplify_quadric_decimation(face_count=TARGET_CELLS)

        X = self._features(work)
        A_S, A_L = self._adjacency(X)

        with torch.no_grad():
            out = model(
                torch.from_numpy(X.T[None, ...]),
                torch.from_numpy(A_S[None, ...]),
                torch.from_numpy(A_L[None, ...]),
            )
        probs = out[0].numpy()
        labels = probs.argmax(axis=-1).astype(np.int32)

        seg = Segmentation(
            mesh=work, labels=labels, probs=probs, jaw=jaw, backend=self.name
        )
        seg.teeth = _build_regions(work, labels, probs, jaw)
        return seg


# ── Region extraction ─────────────────────────────────────────────────────────

# A predicted class covering fewer faces than this is model noise, not a tooth.
# At 10k cells a real tooth is typically 100-800 faces; the floor is set well
# under that so a partially-captured tooth still survives.
_MIN_REGION_FACES = 25


def _build_regions(
    mesh: trimesh.Trimesh, labels: np.ndarray, probs: Optional[np.ndarray], jaw: str
) -> List[ToothRegion]:
    """Turn per-face labels into tooth regions ordered along the arch.

    Arch ordering, not the raw class id, drives FDI assignment: the class id is
    only as trustworthy as the model, whereas the spatial sequence of teeth
    around an arch is a hard anatomical fact.
    """
    areas = mesh.area_faces
    centres = mesh.triangles_center
    regions: List[ToothRegion] = []

    for label in range(1, NUM_TOOTH_CLASSES + 1):
        idx = np.flatnonzero(labels == label)
        if len(idx) < _MIN_REGION_FACES:
            continue
        conf = float(probs[idx, label].mean()) if probs is not None else 0.0
        regions.append(
            ToothRegion(
                label=label,
                face_indices=idx,
                centroid=centres[idx].mean(axis=0),
                area_mm2=float(areas[idx].sum()),
                confidence=conf,
            )
        )

    if not regions:
        return regions

    # Order around the arch by angle about the arch centre in the occlusal
    # plane. Works for a horseshoe regardless of how the scan is posed.
    pts = np.array([r.centroid for r in regions])
    centre = pts[:, :2].mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    order = np.argsort(ang)

    fdi_map = FDI_UPPER if jaw == "upper" else FDI_LOWER
    for position, region_i in enumerate(order):
        r = regions[region_i]
        r.arch_index = position
        # Only meaningful when the model found close to a full arch; with gaps
        # the sequence shifts and the mapping would be confidently wrong.
        if len(regions) >= 10:
            centre_offset = (len(fdi_map) - len(regions)) // 2
            slot = position + centre_offset
            if 0 <= slot < len(fdi_map):
                r.fdi = fdi_map[slot]

    return sorted(regions, key=lambda r: r.arch_index or 0)
