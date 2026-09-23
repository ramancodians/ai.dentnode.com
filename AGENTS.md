<!-- dentnode-platform-context:start -->
## Current DentNode platform context (2026-09-19)

- The verified product and operations knowledge base is the Obsidian vault at `C:\Users\raman\dentnode\knowledge`. Start with `knowledge/Home.md`, then read the relevant architecture/feature note before making domain or infrastructure decisions.
- The authoritative hosting migration record is `knowledge/ops/GCP to VPS Migration Completion — 2026-09-19.md`; infrastructure inventory and retained-resource warnings live in `knowledge/architecture/GCP Infrastructure and Cost.md`.
- `app.dentnode.com` and `ai.dentnode.com` are VPS-only production workloads. Their Cloud Run workflows were removed and the `app-dentnode-com` project currently has zero Cloud Run services.
- Deploy app/AI only through their GitHub Actions VPS workflows: reviewed change → `main` → CI → immutable GHCR digest → restricted VPS dispatcher. Do not recreate Cloud Run or add a second production path unless Raman explicitly requests a rollback/bridge.
- `app.dentnode.com` resolves to VPS `82.112.238.26`; production health endpoints are `/api/health` and `/api/health/ready`. The AI health endpoint is `https://ai.dentnode.com/health`.
- The migration retired hosting, not GCP as a dependency. Never infer permission to delete or replace Cloud SQL, Cloud Storage, Artifact Registry images, Vertex/Google APIs, service accounts, static IPs, backups, or other retained resources.
- When older Markdown conflicts with this block, treat the dated migration note and live repository workflows as authoritative, then update the stale documentation in the same change.
<!-- dentnode-platform-context:end -->

# ai.dentnode.com

This repository owns the Python Laby/AI service now hosted on the DentNode VPS. Keep the service stateless with respect to DentNode tenant data, preserve internal-key boundaries, keep model calls metered, and use the VPS deployment contract. Cloud Run files are historical only and must not be reintroduced as a normal deployment path.

## Shared Jev decision agent

Read `docs/jev-decision-agent.md` and the workspace's `knowledge/architecture/Jev Decision Architecture.md` before changing AI model routing or text matching. Jev is the first model to consider for bounded, non-generative semantic decisions shared across Laby and D10: intent routing, classification, and ranking candidates that a tenant-scoped tool has already retrieved. Ask narrow typed `choice`, `score`, or `noul` questions through OpenRouter's System One API; Jev is not a Chat Completions text-generation model. Keep this decision capability reusable rather than embedding it in one feature workflow.

Use ordinary code and database queries for exact IDs, dates, numbers, authorization, and tenant filtering. The authoritative patient/case/doctor search remains in the app or D10 backend; Jev may help choose among its bounded results but must not invent an identity, expand the search scope, or turn an ambiguous patient match into an automatic action. Preserve the existing user-choice and explicit-confirmation gates for patient calls and other side effects.

Use DeepSeek when a task needs generated text or tool-calling conversation, or when a bounded decision cannot be completed by Jev and a fallback is appropriate. Keep vision, transcription, and speech on capable modality-specific models. Meter each model request and compare observed cost, latency, and quality before claiming savings. Send only the minimum tenant-scoped candidate fields needed for a decision; patient information sent through OpenRouter is external processing and must follow the existing privacy and access rules.
