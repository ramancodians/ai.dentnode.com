# DentNode AI Microservice (ai.dentnode.com)

AI brain for the DentNode ecosystem — RAG chat, vision scanning, webhook-driven sync, token quotas.

## Architecture

- **Runtime**: Bun
- **Framework**: Elysia (TypeScript)
- **Database**: PostgreSQL + pgvector (vector embeddings)
- **ORM**: Prisma 7 with PrismaPg adapter
- **AI Models**: Google Gemini (chat, embeddings, vision) + OpenAI
- **Validation**: Zod

### Folder Layout

- `src/routes/` — API endpoints: `chat.ts`, `scan.ts`, `webhooks.ts`, `admin.ts`, `agents.ts`
- `src/services/` — Core logic: `gemini.ts`, `agent.ts`, `embedder.ts`, `tokenTracker.ts`, `scrubber.ts`
- `src/middleware/auth.ts` — JWT + internal key auth
- `src/db/client.ts` — Prisma singleton
- `prisma/schema.prisma` — Models: `AiTokenLedger`, `LabQuota`, `VectorEmbedding`
- `scripts/` — DB init, RAG population, sync scripts

## Build & Dev

```bash
npm run dev                    # Bun watch mode
npm start                      # Production (no watch)
npm run typecheck              # TypeScript check
npm run db:init                # Initialize database
npm run sync:prod-to-ai        # Sync production data for RAG
npm run sync:prod-to-ai:fresh  # Full reset + sync + verify
npm run db:reset:ai            # Reset AI tables
```

## API Endpoints

| Route | Purpose |
|-------|---------|
| `POST /chat/ask` | RAG-powered lab assistant (streaming) |
| `POST /scan/work-rx` | Vision: parse handwritten WorkRx images → JSON |
| `POST /webhooks/sync` | Entity sync from app.dentnode.com (HMAC-signed) |
| `GET /admin/quota` | Lab token usage & limits |
| `POST /agents/run` | Autonomous agent workflows |

## Conventions

- **Auth**: `x-ai-internal-key` header (service-to-service) OR JWT from app.dentnode.com
- **Multi-tenant**: All data scoped by `labId` via `x-lab-id` header
- **Response pattern**: `{ success: true, data }` / `{ success: false, error }`
- **Webhook security**: HMAC-SHA256 signature validation (`x-webhook-signature`)
- **PII scrubbing**: `scrubber.ts` masks sensitive data before vector storage
- **Token tracking**: All AI calls logged to `AiTokenLedger` with cost tracking
- **Quota enforcement**: Per-lab monthly caps in `LabQuota`; hard 429 on exceeded

## Key Docs

- See `AI_SERVER_GUIDE.md` for external Ollama/GPU server setup
- See `plan.md` for roadmap phases
