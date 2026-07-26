# ai.dentnode.com — Laby ADK Agent

The reasoning service behind **Laby**, DentNode's in-app AI co-pilot for dental
labs. Built on **Google's Agent Development Kit (ADK)**, with every model served
through **OpenRouter** (via ADK's LiteLLM wrapper), deployed to **Cloud Run**
(internal ingress only).

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
| `POST /scan-review` | Oral Scanner AI Review over arch renders | `LABY_VISION_MODEL` |
| `POST /rejected-cases-report` | Rejected-cases ops report (Node cron) | `LABY_VISION_MODEL` |
| `POST /product-update-email` | Weekly product-update marketing copy (Node cron) | `LABY_MODEL` |
| `GET /health` | Liveness/readiness (no auth) | — |

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

CI/CD only (GitHub Actions → Cloud Run, region `asia-south2`, `--ingress
internal`). See `.github/workflows/`. Never deploy manually.
