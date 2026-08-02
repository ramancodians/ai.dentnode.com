# Changelog

All notable changes to the Laby ADK agent service (`ai.dentnode.com`).

Newest first. Each entry records what changed, plus anything that must be true in
the environment for it to run — this service is deployed to Cloud Run by CI, so
missing env vars and Secret Manager entries are the usual cause of a failed rollout.

## [Unreleased] — Scan Review never detected disconnected shells

**Fixed**

- `networkx` added to `requirements.txt`. It is **required, not optional**:
  `trimesh.graph.connected_components` needs a graph engine (scipy or networkx)
  and raises `ImportError("no graph engines available!")` without one. The image
  installed neither.
- The call in `scan_review/geometry.py` sat inside a bare `except Exception:
  pass`, so that ImportError was swallowed whole. `shells` stayed at its default
  `1` and `fragments` stayed `[]` — meaning **disconnected shells and scan
  islands were never detected**, and a scan with floating debris came back
  clean. A false negative in a QA product. The handler still degrades rather
  than failing the report, but now logs a warning with a traceback.
- `trimesh` pinned to `<6.0.0`. The bare `>=4.4.0` resolved to **5.0.0** in CI
  and in the image while development ran on 4.11.2, so a major version was
  reaching production untested.

Caught by CI, not locally: this machine has scipy and networkx installed
globally, so `connected_components` worked here and the suite passed 242/242.
The clean CI environment is what exposed it. Verified afterwards by running the
suite *inside the built image* (trimesh 5.0.0, networkx 3.6.1, numpy 2.5.1) —
242 passed, including `test_floating_fragment_is_detected`.

## [Unreleased] — Scan QA ships (Dockerfile fix)

**Fixed**

- `Dockerfile` never copied `scan_qa`, but `server.py` imports it
  unconditionally at module scope. Any image built from this tree died on
  startup with `ModuleNotFoundError: No module named 'scan_qa'` — before the
  health check could run, so CI's readiness poll would have failed the rollout
  and the canary would have rolled back. Added `COPY scan_qa ./scan_qa`.

**Verified before deploy** (local build of this exact Dockerfile):

- `pytest tests/` — 242 passed.
- Container boots clean, no errors or warnings in startup logs.
- `GET /health` → 200, `GET /scan-qa/health` → 200, `GET /scan-review/health` → 200.

**Known limitation — segmentation is inactive in production.**
`/scan-qa/health` reports `"weights_present":{"Max":false,"Man":false}`. The
image ships neither `torch` (absent from `requirements.txt`) nor
`vendor/meshsegnet/*.zip` (not copied by the Dockerfile), so
`MeshSegNetSegmenter` cannot load and every request degrades to geometry-only
regardless of the `segment` flag. Mesh-level findings — holes, islands, spikes,
winding — are unaffected; per-tooth findings will not appear. Turning
segmentation on is not just a `COPY`: it needs torch (~800 MB installed) plus
the 57 MB weights, and the service currently runs on `--memory=512Mi`, so it
would need a memory bump and a much larger image.

## [Unreleased] — Cloud Run env is now the full config surface

**Changed**

- The CI deploy step now sets **every** variable read by `agent/config.py` and
  `scan_review/config.py` (20 literals), not just the 4 it carried before.
  `--set-env-vars` replaces the whole literal-env list, so any variable missing
  from the workflow was silently deleted from each new revision and the service
  ran on code defaults. Anything the deployment should control now lives in the
  workflow. `PORT` stays out — Cloud Run injects it and rejects it as an input.
- `SCAN_REVIEW_ALLOWED_HOSTS=storage.googleapis.com` is now set in production.
  This is the SSRF control for the only module that dereferences a
  caller-supplied URL; unset, it fell back to "any public host", leaving just
  the private-IP checks. Scans live in the `dentnode-uploads` GCS bucket, so a
  single host covers the real traffic. Widen the list before pointing scan
  review at a new CDN or bucket host.

**Ops note**

- Service traffic is *pinned to a named revision* by the workflow's
  `update-traffic --to-revisions=...` step, so it does not follow "latest".
  A `gcloud run services update` therefore creates a Ready revision that serves
  **0%** until it is explicitly promoted. Applied by hand on 2026-08-02:
  `laby-agent-00015-2jx` carries the full env and is at 100%; `/health` returns
  `openrouter/deepseek/deepseek-v4-flash`.

## [Unreleased] — Scan QA: HTTP surface

- `POST /scan-qa/analyze` + `GET /scan-qa/health`. Takes mesh URLs with a `role`
  per file (prepared_arch / opposing_arch / bite) and returns the QA report with
  every finding's `location` / `locations`. Reuses `scan_review.fetch` for the
  SSRF-guarded download rather than dereferencing URLs itself; bytes are staged
  in a temp dir that is removed on the way out.
- `segment` and `narrate` are per-request flags. Segmentation is ~10s/arch on CPU
  and unreliable on prepared arches, so a caller that only needs mesh-level
  findings can skip it; only narration is metered, since geometry and
  segmentation are local CPU work with no per-call vendor cost.
- A missing segmenter (no weights, no torch) degrades to geometry-only instead
  of failing the request.

Verified against the real 4-file case: HTTP 200, 14.2s geometry-only, and **200
of 200 markers land inside their mesh's bounds**, median 0.03–0.18 mm from the
nearest vertex. The DN3D viewer applies no centering transform to loaded
geometry, so these are usable as world positions unchanged.

## [Unreleased] — Scan QA: findings carry 3-D locations

**Added**

- `check_holes` — `scan_qa` measured holes but never raised a finding for them;
  they only sat in the geometry dict. Now a real check, CRITICAL above 25 mm²
  (a void that size is unmeasured surface, not something to fill).
- `Location` on every `Finding` — mesh coordinates in mm, plus up to 25 marker
  points per finding (largest holes / fragments / sharpest spikes). Populated
  for holes, scan islands and spikes.

Coordinates are what make the DN3D viewer integration worth building: instead of
reading "1,616 holes", a technician clicks the finding and the camera flies to
the worst one. See the DN3DViewer `scan-qa` feature in app.dentnode.com.

## [Unreleased] — Scan QA: segmentation-conditioned scan review (prototype)

New `scan_qa/` package. Hybrid pipeline: a 3D model segments the mesh, then
**deterministic geometry decides**. No network is asked to output "good scan" /
"bad scan" — the LLM only writes prose over findings geometry already made.

`STL → cleanup → segmentation → per-tooth + arch QC → score → LLM explanation`

**Added**

- `scan_qa/segmentation.py` — `Segmenter` protocol so the backbone is swappable
  (MeshSegNet now, PTv3 once DentNode has labelled scans). `MeshSegNetSegmenter`
  runs the MIT-licensed published weights, vendored under `vendor/meshsegnet/`,
  on CPU in ~10s per arch. Preprocessing reproduces `step5_predict.py` exactly
  (15 channels, 10k cells, A_S/A_L at 0.1/0.2) but on trimesh, dropping the
  vedo/VTK dependency.
- `scan_qa/checks.py` — deterministic checks with stated thresholds: scan
  islands, surface spikes, stitching stretch, excess soft tissue, adjacent-tooth
  support, opposing-arch sufficiency, bite coverage, per-tooth capture.
- `scan_qa/pipeline.py` — orchestration, role-aware. Reuses
  `scan_review.geometry` for mesh-level measurement rather than duplicating it.
- `scan_qa/narrate.py` — LLM layer, forbidden from inventing defects or moving
  the score. Never raises; a model outage returns the measurements unchanged.

**Scoring is role-weighted, not a minimum.** Prepared arch 0.5 / opposing 0.2 /
bite 0.3, and repeated findings of one type are capped. A minimum-across-files
rule scored a real case (2 flawless arches + 2 shredded buccal bites) at
**0/100**; the same case now scores 53.2 HIGH, which is the honest reading.

**⚠ KNOWN DOMAIN GAP — do not ship per-tooth output on prep cases yet.** The
published MeshSegNet weights are trained on 72 scans of *natural* dentition.
Controlled test on one case: the opposing natural arch segmented cleanly, while
the **prepared quadrant was largely labelled gingiva** — a prepared tooth has
had its anatomy cut away and does not look like a tooth to the model. This is
exactly the region the QC checks care about. Mesh-level checks are unaffected.

Not yet implemented, and not implementable from segmentation alone: margin
clarity, undercuts, clearance. See the notes in `checks.py`.

## [Unreleased] — Scan Review: dedicated model + prompt size cap

**Changed**

- Scan Review (vision) now reads its own `SCAN_REVIEW_VISION_MODEL` instead of
  the shared `LABY_VISION_MODEL`. Both scan-review legs are set to
  `qwen/qwen3-vl-32b-instruct`; `case_from_image` and `rejected_cases` stay on
  `google/gemini-2.5-flash`. The knob is separate because scan review is tuned
  and evaluated against dental arch renders — swapping its model must not
  silently change unrelated vision features. Falls back to `LABY_VISION_MODEL`
  when unset, so an existing deploy is unaffected until the variable is set.

**Fixed**

- The mesh narrative prompt serialised every hole and fragment record. A real
  4-file case (2 buccal bites, 3,436 holes) produced a **335k-token** prompt and
  OpenRouter rejected it with a 400 against qwen's 131k limit — the review
  silently degraded to geometry-only. Per-record lists are now capped at the 20
  largest, with the true counts and an `holes_omitted` marker retained, cutting
  the geometry payload **98%** (~159k → ~3k tokens). The API response still
  returns every hole; only the prompt is trimmed.

## [Unreleased] — Scan Review: mesh QA from raw STL URLs

New standalone module in [`scan_review/`](./scan_review). **Not part of Laby** —
no ADK, no agent, no tools; a plain fetch → measure → narrate pipeline that
happens to live in the same service. Scaffolded for active iteration.

**Added**

- `POST /scan-review/analyze` — takes raw mesh file URLs (STL/PLY/OBJ/OFF/GLB/3MF), downloads them, and returns a QA review: interior holes (area, perimeter, diameter, position), non-manifold edges, floating fragments, winding consistency, degenerate/duplicate faces, shell count, bounding box, watertightness.
- `GET /scan-review/health` — module config/readiness.
- `scan_review/geometry.py` — deterministic measurement, findings, and a 0–100 score. Two domain behaviours worth knowing: STL triangle soup is vertex-merged before analysis (without it every edge reads as a boundary), and the largest boundary loop of an open scan is classified as its expected `trim_boundary` rather than a defect. Pass `expect_watertight: true` per file for dies/print-bound models to count every opening.
- `scan_review/fetch.py` — SSRF-guarded downloader. See the deployment note below.
- `trimesh>=4.4.0` + `numpy>=1.26.0` in requirements. Core loaders/topology only — the scipy/rtree/embree extras are deliberately not installed.
- `tests/test_scan_review_mesh.py` — 64 tests over synthesised binary STLs, including hole detection, the trim-boundary rule, the STL-header-says-"solid" trap, and the SSRF address matrix.

**Design**

- **Geometry decides, the model describes.** `overall_score` and `risk_level` are computed from the mesh, so the same STL always yields the same verdict. The model writes the prose and may *raise* risk, never lower it. A model failure degrades to measurements + a generated summary rather than failing the request.
- Score is capped at 65 whenever a CRITICAL condition is present, so the number can never read "minor issues" next to a CRITICAL finding.
- Metered as feature `scan_review_mesh` — distinct from Laby's `scan_review` so cost is attributable. A `use_llm: false` run calls no model and creates no billable event.

**Naming**

`POST /scan-review` (flat) remains Laby's **vision** review over rendered arch
images (`agent/scan_review.py`). The new module owns the `/scan-review/*`
sub-namespace. Same name, unrelated code — check which one you mean.

**Deployment requirements**

- **This is the service's only SSRF surface.** It is the one component that dereferences a caller-supplied URL. On Cloud Run, an unguarded fetch of `169.254.169.254` returns a service-account token. The guard enforces https-only, a public-IP requirement on every DNS answer, connection pinning to the validated IP (DNS-rebinding defence), re-validated redirects, and a streaming size cap.
- **Set `SCAN_REVIEW_ALLOWED_HOSTS`** to the bucket/CDN that holds scans before this is exposed to real traffic. The IP-based checks are the fallback, not the intended control.
- `SCAN_REVIEW_ALLOW_INSECURE_FETCH` must stay `false` outside local dev; the service logs a startup WARNING when it is on.
- Image size grows by ~30 MB (trimesh + numpy).

**Status:** in the working tree — **not committed, not deployed.** No Node-side
caller exists yet; nothing invokes the endpoint in production.

## [Unreleased] — AI consolidation: every platform LLM call now runs here

**Added**

- `POST /insights` — clinic metrics report (`agent/insights.py`). Was Gemini in Node.
- `POST /rx-review` — prescription review, 3 parallel sub-agents (`agent/rx_review.py`). Was Gemini in Node.
- `POST /case-from-image` — extract structured case JSON from a photographed lab form (`agent/case_from_image.py`). Was Gemini Vision in Node.
- `POST /scan-review` — Oral Scanner AI Review over rendered arch views (`agent/scan_review.py`). Was NVIDIA Llama Vision / Gemini in Node.
- `POST /rejected-cases-report` — rejection-pattern ops report for a lab (`agent/rejected_cases.py`). Was a Gemini call in a Node cron.
- `POST /product-update-email` — weekly marketing copy from git commits (`agent/marketing_copy.py`). Was OpenAI gpt-4 in a Node cron.
- `agent/usage.py` — central metering client. After every call, posts tokens + real OpenRouter cost to Node's `POST /internal/ai-usage` (`AiUsageEvent` ledger). Fire-and-forget and never raises, so metering cannot break a response.
- `agent/openrouter.py` — shared one-shot chat-completion helper. Requests `usage: {include: true}` so responses carry exact USD cost.
- `LABY_VISION_MODEL` env var (default `google/gemini-2.5-flash`) for image-input features; the tool-calling text model cannot see images.
- `PLATFORM_LAB_ID = "__platform__"` sentinel in `server.py`. The product-update email has no owning lab; its spend is booked to this reserved id so it still appears in the ledger. **Exclude it from per-lab billing and quota queries.**

**Changed**

- Laby chat turns are now metered from `agent/runner.py` via ADK `event.usage_metadata`.
- Scan review sends every arch render as its own image. OpenRouter accepts multiple images per message, which retired Node's `montage.ts` grid workaround (built for NVIDIA's 1-image-per-prompt cap). Per-view labels moved from interleaved text into a single ordered legend, because OpenRouter parses best with all text before images.

**Deployment requirements**

- **`LABY_OPENROUTER_API_KEY` must exist in GCP Secret Manager** (project `app-dentnode-com`). The deploy workflow references it via `--set-secrets`. Absent → `gcloud run deploy` fails with `Secret .../versions/latest was not found`, and the rollback step fails the same way.
- Deploy **this service before** `app.dentnode.com`. The Node backend proxies to these endpoints; shipping the backend first makes every AI feature fail against a service that lacks the routes.
- Cost note: scan review now sends ~12 full-size images instead of 1 composited tile, so its per-call cost rises. `AiUsageEvent.meta.views` records the image count per call.

**Status:** pushed to `main`; **not yet deployed** — blocked on the Secret Manager entry above.
