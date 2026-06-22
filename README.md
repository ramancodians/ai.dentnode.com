# ai.dentnode.com — Laby ADK Agent

The reasoning service behind **Laby**, DentNode's in-app AI co-pilot for dental
labs. Built on **Google's Agent Development Kit (ADK)** with **DeepSeek**
(`deepseek-chat` / V3, via ADK's LiteLLM wrapper), deployed to **Cloud Run**
(internal ingress only).

> Model note: Laby relies on function calling, so it uses `deepseek-chat` (V3).
> Do **not** switch to `deepseek-reasoner` (R1) — it does not support tools.

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
cp .env.example .env          # fill INTERNAL_API_KEY, DEEPSEEK_API_KEY, NODE_INTERNAL_BASE_URL
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
