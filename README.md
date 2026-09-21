# ai.dentnode.com — Laby ADK Agent

The reasoning service behind **Laby**, DentNode's in-app AI co-pilot for dental
labs. Built on **Google's Agent Development Kit (ADK)**, with every model served
through **OpenRouter** (via ADK's LiteLLM wrapper), deployed privately on the
DentNode VPS behind Caddy.

> **OpenRouter only.** This service never calls a model provider's API
> directly. Provider credentials are configured as **BYOK** keys inside the
> OpenRouter dashboard, so the service ships a single `OPENROUTER_API_KEY` and
> OpenRouter routes the request to the provider. Do not add
> `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` or any other
> provider key here.

> Model note: `LABY_MODEL` takes an OpenRouter model id (default
> `deepseek/deepseek-v4-flash`); the `openrouter/` litellm route prefix is added
> automatically in `agent/config.py`. Laby relies on function calling, so the
> model **must** support tools — do not switch to `deepseek/deepseek-r1`.
> Switching providers is now a one-line env change (`LABY_MODEL=openai/gpt-5`,
> `anthropic/claude-sonnet-5`, …) with no code change.

> The previous Bun/TypeScript experiment (pgvector RAG + WorkRx scanner +
> automation agents) is archived under [`legacy/`](./legacy) for reference.

## Architecture

```
Browser ──SSE──> Node backend (app.dentnode.com)  [auth, history, rate-limit, ₹ cap]
                      │  POST /agent/run  (x-internal-key, NDJSON stream)
                      ▼
              THIS service (Laby ADK agent)
                      │  curated FunctionTools
                      ▼
              Node /api/internal/laby-tools/:tool  (x-internal-key) ──> Prisma/MySQL
```

### Endpoints

All are internal-only (`x-internal-key`), and every one meters its LLM spend to
Node's `AiUsageEvent` ledger via `POST /api/internal/ai-usage`.

| Endpoint | Feature | Model |
|---|---|---|
| `POST /agent/run` | Laby chat co-pilot (ADK tool loop, NDJSON stream) | `LABY_MODEL` |
| `POST /insights` | Clinic metrics report | `LABY_MODEL` |
| `POST /rx-review` | Prescription review (3 sub-agents) | `LABY_MODEL` |
| `POST /case-from-image` | Extract case JSON from a form photo | `LABY_VISION_MODEL` |
| `POST /image-to-entry` | Convert a form photo to a DentNode entry-create draft (no write) | `LABY_VISION_MODEL` |
| `POST /scan-review` | Oral Scanner AI Review over arch renders | `LABY_VISION_MODEL` |
| `POST /rejected-cases-report` | Rejected-cases ops report (Node cron) | `LABY_VISION_MODEL` |
| `POST /product-update-email` | Weekly product-update marketing copy (Node cron) | `LABY_MODEL` |
| `POST /audio-to-text` | **Audio-to-Text** — transcribe (with speaker diarization) + summarise a Digital Ocean audio URL (shared by app.dentnode.com + d10.live) | `AUDIO_TO_TEXT_MODEL` |
| `POST /internal/call-audio/analyze` | **Call analysis** — transcribe + summarise an app-owned multipart recording upload | `AUDIO_TO_TEXT_MODEL` |
| `POST /text-to-speech` | **Text-to-Speech** — synthesise speech and stream the audio bytes back (open to every trusted internal caller) | `TEXT_TO_SPEECH_MODEL` |
| `GET /text-to-speech/voices` | Voices, formats and limits the TTS endpoint accepts | — |
| `POST /scan-review/analyze` | **Scan Review** — mesh QA from raw STL URLs (standalone module, not Laby) | `SCAN_REVIEW_MODEL` |
| `GET /scan-review/health` | Scan Review config/readiness (no auth) | — |
| `GET /health` | Liveness/readiness (no auth) | — |

> Note the two similarly-named endpoints. `POST /scan-review` is Laby's
> **vision** review of rendered arch images. `POST /scan-review/analyze` is the
> standalone **geometry** module described below. They share a name and nothing
> else.

Every LLM call in the DentNode platform now goes through this service — the Node
backend no longer depends on any model SDK. `/product-update-email` is the one
call with no owning lab; its spend is booked to the reserved `__platform__`
lab id so it still appears in the ledger. Exclude that id from per-lab billing.

Node stays the tenant boundary: it authenticates the user, derives `lab_id` from
the JWT, fetches any images or DB rows, and passes only the model inputs here.
This service never queries the database and never dereferences a client URL.

The agent has **no database**. Every data lookup is a curated, typed tool that
calls back into the Node backend, which runs the Prisma query scoped to the
caller's `lab_id`. The model never sees or supplies `lab_id` — it is injected
into the ADK session state by Node from the verified JWT, so cross-tenant access
is impossible by construction. There is no raw text-to-SQL.

## Tools (Phase 1)

| Tool | Question it answers |
|------|---------------------|
| `cases_received` | "How many cases did I receive today / this week?" |
| `cases_timeline` | "What does my timeline look like for the next 3 weeks?" |
| `expected_volume` | "How many cases can I expect tomorrow?" (heuristic estimate) |
| `inactive_clients` | "Which clients are not sending me cases?" |
| `product_sales` | "Which products are selling more?" |
| `staff_activity` | "Which staff are not logging in properly?" |

## Audio-to-Text

`agent/audio_to_text.py`. A Digital Ocean audio URL in, a transcript out. This
goes through OpenRouter's `POST /audio/transcriptions` endpoint — a transcription
model, not a chat model with an `input_audio` part — so `AUDIO_TO_TEXT_MODEL`
must be a transcription model. The default, `microsoft/mai-transcribe-2`, also
exposes Azure diarization, which is what fills in `segments[]`.

```json
{
  "transcript": "...",
  "summary": "...",
  "segments": [{ "start": 0.0, "end": 4.2, "text": "...", "speaker": 0 }]
}
```

**`speaker` is recording-local.** Speaker `0` in one recording is not the same
person as speaker `0` in another. It separates voices within a single file and
nothing more — there is no cross-recording voice identity here, so do not store
it as one or join on it.

`summary: true` costs a **second** provider call: the transcription endpoint
returns a transcript only, so the summary is written separately by `LABY_MODEL`.

The URL is caller-supplied, which makes `AUDIO_TO_TEXT_ALLOWED_HOSTS` an SSRF
control rather than a preference — it is pinned in the deploy workflow for that
reason, and `AUDIO_TO_TEXT_ALLOW_INSECURE_FETCH` must stay `false` anywhere
deployed.

### Internal call recording analysis

`POST /internal/call-audio/analyze` is the byte-upload contract used by the
DentNode calling integration. It requires `x-internal-key` and multipart fields
`file`, `lab_id`, and `call_id`; it returns exactly `{"transcript": ..., "summary":
...}`. Only the explicit audio MIME allowlist in `server.py` is accepted, and
the upload is bounded by `AUDIO_TO_TEXT_MAX_BYTES`. The route never fetches a
caller-supplied URL and stores neither audio nor generated text.

This service is stateless with respect to call analysis. The app owns durable
job state and must deduplicate/retry by `call_id`; repeating a request here may
repeat provider work and usage charges. Usage events are attributed to
`lab_id`, with `call_id` used only as the metering correlation id.

## Text-to-Speech

`agent/text_to_speech.py`. Text in, audio bytes out. A single-call feature agent
open to every trusted internal caller — `app.dentnode.com` via `x-internal-key`,
`d10.live` via `x-d10-internal-key`, like `/audio-to-text`.

**It stores nothing.** The audio is streamed straight back; whether it lands in
Spaces, goes out over WhatsApp, or is played once and dropped is the calling
application's decision, because only that application knows the retention rules
for what was said.

```bash
curl -sS -X POST https://<host>/text-to-speech \
  -H "x-internal-key: $INTERNAL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Your crown case is ready for pickup.","voice":"alloy"}' \
  -o speech.wav
```

| Field | Default | Notes |
|---|---|---|
| `text` | — | Required. Capped by `TEXT_TO_SPEECH_MAX_CHARS` (4000). |
| `voice` | `TEXT_TO_SPEECH_VOICE` | See `GET /text-to-speech/voices`. |
| `audio_format` | `wav` | `wav` or `pcm` (raw 24 kHz mono 16-bit LE frames). |
| `stream` | `true` | `false` buffers one file with a `Content-Length`. |
| `lab_id` / `user_id` | `__platform__` | Attribution on the `AiUsageEvent` row. |

The body is binary, so metadata rides in `X-TTS-*` headers. Buffered responses
add `X-TTS-Audio-Ms`, `X-TTS-Cost-Usd` and `X-TTS-Transcript-B64` (base64 so a
non-Latin-1 script cannot break the header); streaming responses cannot, because
none of it is known until the stream drains — read it from the ledger instead.

### Two things about this endpoint are not obvious

**There is no free TTS on OpenRouter, and only one sane-priced option.** Of 430
models in the catalog exactly four emit audio: two Lyria models (music, not
speech) and `openai/gpt-audio` / `openai/gpt-audio-mini`. The mini variant is
26.7x cheaper per audio output token ($0.0000024 vs $0.000064), which measures
out at **~$0.0043 per minute of speech** (₹0.37/min) — a 6.4-second
notification costs $0.00044. `TEXT_TO_SPEECH_MODEL` is therefore a spend knob,
and `TEXT_TO_SPEECH_MAX_CHARS` is the per-call cap.

**MP3 is unreachable, so we emit WAV.** OpenRouter rejects audio output without
`stream: true`, and rejects every container but `pcm16` once streaming. We
receive raw PCM and write the 44-byte RIFF header ourselves — no encoder, no
ffmpeg, nothing added to a 512Mi image. A streaming WAV carries the streaming
sentinel in its two size fields (the length is unknown when the header goes
out); ffmpeg, ffprobe and browsers all read such a file correctly, but anything
that trusts the size field verbatim will report a nonsense duration. Send
`stream: false` when the file's own metadata has to be exact.

Note also that `gpt-audio-mini` is a chat model, not a TTS engine: a system
prompt clamps it to reading rather than answering, and it is the injection
boundary, since the text being spoken is caller data. `X-TTS-Transcript-B64`
reports what the model actually said, so a caller can verify it verbatim.

## Scan Review (standalone module)

Lives in [`scan_review/`](./scan_review) and is **not part of Laby** — no ADK, no
agent session, no tools. It takes raw mesh file URLs, downloads them, measures
the geometry, and returns a QA review.

```
STL URLs ──fetch.py──> bytes ──geometry.py──> measurements + findings + score
                                                        │
                                                        ▼
                                              LLM narrative (prose only)
```

**Geometry decides, the model describes.** `overall_score` and `risk_level` are
computed from the mesh, not asked of a model, so the same STL always produces
the same verdict. The model turns the numbers into something a technician acts
on, and may *raise* risk but never lower it. If the model call fails, the
measurements are still returned.

What it measures: interior holes (count, area, perimeter, diameter, position),
non-manifold edges, floating fragments, winding consistency, degenerate and
duplicate faces, shell count, surface area, bounding box, and watertightness.

Two things it gets right that naive implementations do not:

- **STL is triangle soup.** No vertex sharing — analysed as-is, every edge looks
  like a boundary. Vertices are merged by position first, then re-welded at
  0.1 µm to absorb exporter float noise that otherwise reads as thousands of
  hairline "holes".
- **An intraoral scan is not supposed to be watertight.** It is an open surface
  with a trim boundary around its perimeter. That largest loop is reported
  separately as `trim_boundary` and is not counted as a defect. Pass
  `expect_watertight: true` per file for a die, a designed crown, or anything
  print-bound, and then every opening counts.

```bash
curl -X POST http://localhost:8080/scan-review/analyze \
  -H "x-internal-key: $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "lab_id": "<labId>",
        "case_context": {"product": "Zirconia crown", "tooth": 36},
        "files": [
          {"url": "https://scans.example.com/upper.stl", "label": "Upper arch"},
          {"url": "https://scans.example.com/die.stl", "label": "Die",
           "expect_watertight": true}
        ]
      }'
```

Set `"use_llm": false` for a geometry-only run — deterministic, free, and not
metered (no model call means no billable event).

> ⚠️ **This is the only component in the service that dereferences a
> caller-supplied URL**, which makes it the only SSRF surface. On Cloud Run an
> unguarded fetch of `169.254.169.254` yields a service-account token, so
> `scan_review/fetch.py` enforces https-only, a public-IP requirement on every
> DNS answer, connection pinning to the validated IP (against DNS rebinding),
> re-validated redirects, and a streaming size cap. **Set
> `SCAN_REVIEW_ALLOWED_HOSTS` in production** so it can only reach our own
> storage — the IP checks alone are the weaker guarantee.

## Local development

```bash
cp .env.example .env          # fill INTERNAL_API_KEY, OPENROUTER_API_KEY, NODE_INTERNAL_BASE_URL
pip install -r requirements.txt
python server.py              # serves on $PORT (default 8080)
```

Smoke test (Node backend must be running and reachable):

```bash
curl -N -X POST http://localhost:8080/agent/run \
  -H "x-internal-key: $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"lab_id":"<labId>","user_id":"<userId>","question":"How many cases did I get today?"}'
```

Or run the whole stack with `docker-compose up` from the workspace root.

## Deploy

Production deploys run through `.github/workflows/deploy-vps.yaml`: a push to
`main` is tested, published to GHCR by immutable digest, and released through
the VPS restricted dispatcher. There is no staging deployment.

Cloud Run hosting has been retired. Rollbacks use the previous immutable VPS
image retained by the restricted dispatcher. Do not deploy production outside
the VPS GitHub Actions workflow.
