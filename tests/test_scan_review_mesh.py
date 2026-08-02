"""Tests for the standalone Scan Review module (mesh QA from STL URLs).

Distinct from tests/test_scan_review.py, which covers Laby's *vision* review of
rendered arch images. These two share a name and nothing else.

Meshes are synthesised in-process as real binary STLs so the geometry
assertions run against the actual trimesh pipeline — including the vertex-merge
step, which is the part most likely to silently break hole detection.
"""

import ipaddress
import struct

import numpy as np
import pytest

from scan_review.config import settings as sr_settings
from scan_review.fetch import MeshFetchError, sniff_file_type, validate_url
from scan_review.fetch import _is_public_ip, _resolve_host
from scan_review.geometry import (
    MeshInspectError,
    build_findings,
    inspect_mesh,
    risk_from_findings,
    score_report,
)
from scan_review.review import (
    FileResult,
    _escalate_only,
    build_flags,
    parse_model_json,
)
from tests.conftest import TEST_KEY


# ── Mesh fixtures ─────────────────────────────────────────────────────────────


def binary_stl(triangles) -> bytes:
    """Serialise [(v0, v1, v2), ...] as a binary STL.

    Written as true triangle soup (no shared vertices), exactly as a scanner
    exports it — which is what makes these tests meaningful.
    """
    out = [b"\x00" * 80, struct.pack("<I", len(triangles))]
    for tri in triangles:
        v0, v1, v2 = (np.asarray(v, dtype=np.float32) for v in tri)
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm else np.zeros(3, dtype=np.float32)
        out.append(struct.pack("<3f", *normal))
        for v in (v0, v1, v2):
            out.append(struct.pack("<3f", *v))
        out.append(b"\x00\x00")
    return b"".join(out)


def cube_triangles(size: float = 20.0):
    """Closed 20 mm cube: 12 triangles, watertight, no defects."""
    s = size
    v = [
        (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
        (0, 0, s), (s, 0, s), (s, s, s), (0, s, s),
    ]
    quads = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[d]))
    return tris


def plane_with_hole(n: int = 4, spacing: float = 10.0, punch=(1, 1)):
    """Open n×n-vertex grid with one cell removed.

    This is the shape that matters: an open surface (like a real intraoral
    scan) that has BOTH an outer trim boundary and a genuine interior hole. The
    module must tell those two apart.
    """
    def vert(i, j):
        return (i * spacing, j * spacing, 0.0)

    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            if punch is not None and (i, j) == punch:
                continue
            tris.append((vert(i, j), vert(i + 1, j), vert(i + 1, j + 1)))
            tris.append((vert(i, j), vert(i + 1, j + 1), vert(i, j + 1)))
    return tris


# ── Geometry: the core promise ────────────────────────────────────────────────


def test_closed_cube_is_watertight_with_no_holes():
    report = inspect_mesh(binary_stl(cube_triangles()), "stl", label="cube")
    assert report.faces == 12
    # 8 corners after the merge — proof the triangle-soup merge actually ran.
    assert report.vertices == 8
    assert report.watertight is True
    assert report.open_edges == 0
    assert report.holes == []
    assert report.non_manifold_edges == 0
    assert score_report(report) == 100.0


def test_stl_triangle_soup_is_merged_before_analysis():
    """Without the merge every edge reads as a boundary and nothing works."""
    report = inspect_mesh(binary_stl(cube_triangles()), "stl", label="cube")
    # 12 triangles x 3 vertices = 36 unmerged; must collapse to the 8 corners.
    assert report.vertices == 8
    assert report.open_edges_before_merge == 0


def test_interior_hole_is_found_and_measured():
    """The headline feature: a punched cell in an open surface is a hole."""
    report = inspect_mesh(binary_stl(plane_with_hole()), "stl", label="upper")

    assert len(report.holes) == 1
    hole = report.holes[0]
    # The punched cell is 10 x 10 mm.
    assert hole.area_mm2 == pytest.approx(100.0, rel=0.02)
    assert hole.edges == 4
    assert hole.perimeter_mm == pytest.approx(40.0, rel=0.02)
    assert hole.is_trim_boundary is False


def test_outer_perimeter_is_not_reported_as_a_hole():
    """An intraoral scan is an open surface; its trim edge is normal."""
    report = inspect_mesh(binary_stl(plane_with_hole()), "stl", label="upper")

    assert report.trim_boundary is not None
    assert report.trim_boundary.is_trim_boundary is True
    # The 30 x 30 mm patch outline, not the 10 x 10 mm hole.
    assert report.trim_boundary.perimeter_mm == pytest.approx(120.0, rel=0.02)
    assert all(not h.is_trim_boundary for h in report.holes)


def test_intact_open_surface_reports_zero_holes():
    """No false positives: an open scan with no defect must come back clean."""
    report = inspect_mesh(binary_stl(plane_with_hole(punch=None)), "stl", label="lower")
    assert report.holes == []
    assert report.trim_boundary is not None
    assert score_report(report) == 100.0


def test_expect_watertight_counts_every_opening():
    """For a die or a print-ready model, the trim-boundary exemption is wrong."""
    open_cube = cube_triangles()[:-2]  # drop one face

    auto = inspect_mesh(binary_stl(open_cube), "stl", label="die")
    assert auto.holes == []          # largest loop treated as a trim edge
    assert auto.trim_boundary is not None

    strict = inspect_mesh(
        binary_stl(open_cube), "stl", label="die", expect_watertight=True
    )
    assert len(strict.holes) == 1
    assert strict.trim_boundary is None
    assert strict.watertight is False
    assert score_report(strict) < score_report(auto)


def test_floating_fragment_is_detected():
    """Stray tongue/cheek debris shows up as a disconnected shell."""
    tris = cube_triangles(size=20.0)
    # A tiny separate triangle far from the cube.
    tris.append(((100.0, 100.0, 0.0), (100.2, 100.0, 0.0), (100.0, 100.2, 0.0)))

    report = inspect_mesh(binary_stl(tris), "stl", label="upper")
    assert report.shells == 2
    assert len(report.fragments) == 1
    assert "floating_fragments" in build_flags(
        [FileResult(label="upper", url="u", ok=True, report=report)]
    )


def test_implausible_scale_is_flagged():
    """A metres-instead-of-mm export is the usual cause."""
    report = inspect_mesh(binary_stl(cube_triangles(size=0.02)), "stl", label="tiny")
    findings = build_findings(report)
    dimension = [f for f in findings if f["category"] == "DIMENSIONS"]
    assert len(dimension) == 1
    assert dimension[0]["severity"] == "CRITICAL"
    assert score_report(report) < 80


def test_unparseable_bytes_raise_inspect_error():
    with pytest.raises(MeshInspectError):
        inspect_mesh(b"this is not a mesh at all", "stl", label="bad")


# ── Findings and scoring ──────────────────────────────────────────────────────


def test_findings_are_generated_for_a_hole():
    report = inspect_mesh(binary_stl(plane_with_hole()), "stl", label="upper")
    findings = build_findings(report)

    hole_findings = [f for f in findings if f["category"] == "PRINT_READINESS"]
    assert hole_findings
    # A 100 mm² hole is well over the 1 mm² threshold.
    assert hole_findings[0]["severity"] == "CRITICAL"
    assert all(f["source"] == "geometry" for f in findings)


def test_clean_mesh_still_produces_an_explicit_finding():
    """Silence reads as "not checked"; say so explicitly instead."""
    report = inspect_mesh(binary_stl(cube_triangles()), "stl", label="cube")
    findings = build_findings(report)
    assert len(findings) == 1
    assert findings[0]["severity"] == "INFO"
    assert "No geometry defects" in findings[0]["title"]


def test_score_is_deterministic():
    data = binary_stl(plane_with_hole())
    a = score_report(inspect_mesh(data, "stl", label="x"))
    b = score_report(inspect_mesh(data, "stl", label="x"))
    assert a == b


def test_score_never_contradicts_a_critical_finding():
    """A "minor issues" score (70-89) next to a CRITICAL finding is nonsense.

    The 100 mm² hole in this fixture is a missing region, not a blemish; the
    number the lab sees has to reflect that.
    """
    report = inspect_mesh(binary_stl(plane_with_hole()), "stl", label="upper")
    findings = build_findings(report)
    score = score_report(report)

    assert any(f["severity"] == "CRITICAL" for f in findings)
    assert score <= 65.0
    assert risk_from_findings(findings, score) == "HIGH"


def test_hole_size_drives_the_penalty_not_just_the_count():
    """One large void must cost more than several pinholes."""
    one_large = inspect_mesh(binary_stl(plane_with_hole(n=4)), "stl", label="a")

    # Two pinholes punched from a fine grid: 0.5 mm cells = 0.25 mm² each.
    fine = plane_with_hole(n=30, spacing=0.5, punch=(1, 1))
    fine = [t for t in fine if not _in_cell(t, (5, 5), 0.5)]
    two_small = inspect_mesh(binary_stl(fine), "stl", label="b")

    assert len(two_small.holes) == 2
    assert all(h.area_mm2 < 1.0 for h in two_small.holes)
    # 2 pinholes beat 1 large void.
    assert score_report(two_small) > score_report(one_large)


def _in_cell(tri, cell, spacing):
    """True if every vertex of `tri` belongs to grid cell `cell`."""
    i, j = cell
    lo_x, hi_x = i * spacing, (i + 1) * spacing
    lo_y, hi_y = j * spacing, (j + 1) * spacing
    return all(
        lo_x - 1e-9 <= v[0] <= hi_x + 1e-9 and lo_y - 1e-9 <= v[1] <= hi_y + 1e-9
        for v in tri
    )


def test_risk_mapping():
    assert risk_from_findings([{"severity": "CRITICAL"}], 95) == "HIGH"
    assert risk_from_findings([{"severity": "INFO"}], 40) == "HIGH"
    assert risk_from_findings([{"severity": "WARNING"}], 90) == "MEDIUM"
    assert risk_from_findings([{"severity": "INFO"}], 100) == "LOW"


# ── The model may escalate risk, never lower it ───────────────────────────────


@pytest.mark.parametrize(
    "measured,suggested,expected",
    [
        ("LOW", "HIGH", "HIGH"),
        ("MEDIUM", "HIGH", "HIGH"),
        ("HIGH", "LOW", "HIGH"),       # the important one
        ("MEDIUM", "LOW", "MEDIUM"),
        ("MEDIUM", "nonsense", "MEDIUM"),
        ("HIGH", None, "HIGH"),
    ],
)
def test_escalate_only(measured, suggested, expected):
    assert _escalate_only(measured, suggested) == expected


# ── Parser ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    ['{"summary": "ok"}', '```json\n{"summary": "ok"}\n```', 'Sure:\n{"summary": "ok"}'],
)
def test_parses_json_variants(raw):
    assert parse_model_json(raw) == {"summary": "ok"}


@pytest.mark.parametrize("raw", ["", "no json", "[1,2]", None])
def test_unparseable_returns_none(raw):
    assert parse_model_json(raw) is None


# ── SSRF guard ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1",            # loopback
        "169.254.169.254",      # GCP/AWS metadata — the one that matters
        "10.1.2.3",             # RFC1918
        "192.168.0.5",
        "172.16.4.4",
        "100.64.1.1",           # CGNAT
        "0.0.0.0",
        "::1",
        "fe80::1",
        "::ffff:169.254.169.254",  # IPv4-mapped metadata address
    ],
)
def test_internal_addresses_are_rejected(addr):
    assert _is_public_ip(ipaddress.ip_address(addr)) is False


@pytest.mark.parametrize("addr", ["8.8.8.8", "1.1.1.1", "2606:4700::1111"])
def test_public_addresses_are_allowed(addr):
    assert _is_public_ip(ipaddress.ip_address(addr)) is True


def test_http_is_rejected_by_default():
    with pytest.raises(MeshFetchError, match="https"):
        validate_url("http://example.com/scan.stl")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/scan.stl",
        "ftp://example.com/scan.stl",
    ],
)
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(MeshFetchError):
        validate_url(url)


def test_embedded_credentials_are_rejected():
    with pytest.raises(MeshFetchError, match="credentials"):
        validate_url("https://user:pass@example.com/scan.stl")


def test_empty_url_is_rejected():
    with pytest.raises(MeshFetchError):
        validate_url("")


def test_localhost_resolution_is_refused():
    """The DNS check, not just the literal-IP check."""
    with pytest.raises(MeshFetchError, match="non-public"):
        _resolve_host("localhost", 443)


def test_host_allowlist_blocks_other_hosts(monkeypatch):
    monkeypatch.setattr(sr_settings, "allowed_hosts_raw", "scans.dentnode.com")
    with pytest.raises(MeshFetchError, match="ALLOWED_HOSTS"):
        validate_url("https://evil.example.com/scan.stl")
    # The allowlisted host still passes.
    scheme, host, port, _url = validate_url("https://scans.dentnode.com/a.stl")
    assert (scheme, host, port) == ("https", "scans.dentnode.com", 443)


def test_host_allowlist_supports_suffix_entries(monkeypatch):
    monkeypatch.setattr(sr_settings, "allowed_hosts_raw", ".dentnode.com")
    scheme, host, _port, _url = validate_url("https://cdn.dentnode.com/a.stl")
    assert host == "cdn.dentnode.com"
    with pytest.raises(MeshFetchError):
        validate_url("https://dentnode.com.evil.io/a.stl")


# ── Format sniffing ───────────────────────────────────────────────────────────


def test_binary_stl_is_identified_by_length_arithmetic():
    data = binary_stl(cube_triangles())
    assert sniff_file_type(data, "https://x/y", None) == "stl"


def test_binary_stl_whose_header_says_solid_is_still_binary():
    """The classic STL trap: a binary file whose 80-byte header starts 'solid'."""
    data = binary_stl(cube_triangles())
    spoofed = b"solid exported-by-scanner" + data[25:]
    assert len(spoofed) == len(data)
    assert sniff_file_type(spoofed, "https://x/y.stl", None) == "stl"


def test_ascii_stl_is_identified():
    ascii_stl = b"solid test\nfacet normal 0 0 1\nendsolid test\n"
    assert sniff_file_type(ascii_stl, "https://x/y", None) == "stl"


def test_ply_and_off_are_identified():
    assert sniff_file_type(b"ply\nformat ascii 1.0\n", "https://x/y", None) == "ply"
    assert sniff_file_type(b"OFF\n8 6 0\n", "https://x/y", None) == "off"


def test_obj_falls_back_to_the_extension():
    assert sniff_file_type(b"v 0 0 0\nv 1 0 0\n", "https://x/model.obj", None) == "obj"


def test_unknown_content_is_rejected():
    with pytest.raises(MeshFetchError, match="not a recognised mesh"):
        sniff_file_type(b"<html>nope</html>", "https://x/y.html", "text/html")


# ── Endpoint ──────────────────────────────────────────────────────────────────

_BODY = {
    "lab_id": "lab-1",
    "files": [{"url": "https://scans.example.com/upper.stl", "label": "Upper"}],
}


def test_missing_key_returns_401(client):
    assert client.post("/scan-review/analyze", json=_BODY).status_code == 401


def test_missing_files_returns_422(client):
    resp = client.post(
        "/scan-review/analyze",
        json={"lab_id": "lab-1", "files": []},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 422


def test_too_many_files_returns_400(client, monkeypatch):
    monkeypatch.setattr(sr_settings, "max_files", 2)
    resp = client.post(
        "/scan-review/analyze",
        json={
            "lab_id": "lab-1",
            "files": [
                {"url": f"https://scans.example.com/{i}.stl"} for i in range(3)
            ],
        },
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 400
    assert "Too many files" in resp.json()["error"]


def test_health_reports_module_config(client):
    resp = client.get("/scan-review/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["module"] == "scan-review"
    assert body["status"] == "healthy"


def test_laby_vision_scan_review_route_still_exists(client):
    """The new /scan-review/* sub-namespace must not shadow the flat route."""
    # A valid body for Laby's vision endpoint reaches its auth check and is
    # rejected with 401 — proving the route still resolves rather than 404ing.
    resp = client.post("/scan-review", json={"lab_id": "lab-1", "case_context": {}, "views": []})
    assert resp.status_code == 401


def test_returns_full_payload(client, monkeypatch):
    """End-to-end shape, with the model layer stubbed out."""
    import scan_review.router as router_mod

    async def _fake_llm(**kwargs):
        raise AssertionError("should not be reached")

    real_files = [
        FileResult(
            label="Upper",
            url="https://scans.example.com/upper.stl",
            ok=True,
            report=inspect_mesh(binary_stl(plane_with_hole()), "stl", label="Upper"),
            findings=[],
            score=61.0,
        )
    ]

    from scan_review.review import ScanReviewResult

    async def _mock(**kwargs):
        assert kwargs["today"] == "2026-07-31"
        return ScanReviewResult(
            summary="One hole on the upper arch.",
            risk_level="HIGH",
            overall_score=61.0,
            findings=[{"id": "g1", "severity": "CRITICAL", "source": "geometry"}],
            flags=["holes", "large_holes"],
            files=real_files,
            model="deepseek/deepseek-v4-flash",
            parsed=True,
            usage={"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100},
            cost_usd=0.0004,
            latency_ms=4100,
        )

    monkeypatch.setattr(router_mod, "generate_scan_review", _mock)
    monkeypatch.setattr(router_mod, "_report", lambda body, result: None)

    resp = client.post(
        "/scan-review/analyze",
        json={**_BODY, "today": "2026-07-31"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["risk_level"] == "HIGH"
    assert body["overall_score"] == 61.0
    assert body["flags"] == ["holes", "large_holes"]
    assert body["analyzed_count"] == 1
    assert body["requested_count"] == 1
    assert body["files"][0]["geometry"]["hole_count"] == 1


def test_metering_records_geometry_context(client, monkeypatch):
    """Cost analysis later needs file count and outcome, not just tokens."""
    import scan_review.router as router_mod
    from scan_review.review import ScanReviewResult

    captured = {}

    async def _mock(**kwargs):
        return ScanReviewResult(
            summary="ok",
            risk_level="LOW",
            overall_score=100.0,
            findings=[],
            flags=["clean"],
            files=[FileResult(label="Upper", url="u", ok=True, score=100.0)],
            model="deepseek/deepseek-v4-flash",
            parsed=True,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost_usd=0.00001,
            latency_ms=900,
        )

    # Capture synchronously at call time — metering is fire-and-forget, so
    # asserting inside the coroutine would race the response.
    def _capture(**kwargs):
        captured.update(kwargs)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(router_mod, "generate_scan_review", _mock)
    monkeypatch.setattr(router_mod, "report_usage", _capture)

    resp = client.post(
        "/scan-review/analyze", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )
    assert resp.status_code == 200
    assert captured["feature"] == "scan_review_mesh"
    assert captured["lab_id"] == "lab-1"
    assert captured["cost_source"] == "openrouter"
    assert captured["meta"]["files"] == 1
    assert captured["meta"]["score"] == 100.0


def test_geometry_only_run_is_not_metered(client, monkeypatch):
    """A run with no model call must not create a billable event."""
    import scan_review.router as router_mod
    from scan_review.review import ScanReviewResult

    called = {"metered": False}

    async def _mock(**kwargs):
        assert kwargs["use_llm"] is False
        return ScanReviewResult(
            summary="geometry only",
            risk_level="LOW",
            overall_score=100.0,
            findings=[],
            flags=["clean"],
            files=[FileResult(label="Upper", url="u", ok=True, score=100.0)],
            model=None,          # no model ran
            llm_error=None,
        )

    def _capture(**kwargs):
        called["metered"] = True

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(router_mod, "generate_scan_review", _mock)
    monkeypatch.setattr(router_mod, "report_usage", _capture)

    resp = client.post(
        "/scan-review/analyze",
        json={**_BODY, "use_llm": False},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert called["metered"] is False
