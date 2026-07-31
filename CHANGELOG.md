# Changelog

All notable changes to the Laby ADK agent service (`ai.dentnode.com`).

Newest first. Each entry records what changed, plus anything that must be true in
the environment for it to run — this service is deployed to Cloud Run by CI, so
missing env vars and Secret Manager entries are the usual cause of a failed rollout.

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
