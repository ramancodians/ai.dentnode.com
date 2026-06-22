# DentNode AI Service (ai.dentnode.com) Product & Engineering Plan

## 1. Executive Summary
The goal is to build a modern, blisteringly fast, dedicated AI microservice (`ai.dentnode.com`) that acts as the intelligent "Brain" for the DentNode ecosystem. It will securely host AI features, track usage cleanly by lab, and run fully decoupled from the core monolithic `app.dentnode.com` application. 

By separating this into a Bun-powered microservice, we ensure high throughput with low latency, providing immediate real-time AI capabilities without hogging resources on the main production database or server.

---

## 2. Product Vision & Owner Use Cases

Acting as a product manager, here is how the AI must serve the target persona—a **Dental Lab Owner** or **Lab Manager**:

### A. The "Omniscient" Lab Assistant (Chat & Insight)
*   **Query Operations:** "Which cases are due for Dr. Smith tomorrow but are still in the porcelain department?"
*   **Financial Insights:** "Summarize my overdue payments and draft polite reminder WhatsApp messages."
*   **Quality Control:** "Which technician had the most remakes this quarter, and on what materials?"
*   The AI Assistant will feel like a senior manager who has complete working memory of the lab's operations via a synchronized Vector Database.

### B. Smart WorkRx (Prescription) Scanner
*   **Use Case:** A physical prescription arrives with a mould. Simply snap a photo or scan it.
*   **AI Action:** The AI Vision model reads the handwritten or printed RX, extracts tooth numbers, identifies the quadrant (e.g., UR6, LR7), determines the requested product (e.g., Zirconia Crown, PFM), and isolates specific shade instructions (e.g., "A2 with translucent incisal edge").
*   **Benefit:** Reduces manual data entry time from 3 minutes per case to 10 seconds.

### C. Automated Agents (Automation Tasks)
*   **Use Case:** Background AI workers that constantly monitor the system.
*   **Examples:**
    *   If an intraoral scan is uploaded and the margin is unclear, an Agent automatically drafts an email to the clinic asking for clarification before manufacturing begins.
    *   If a high-value case is stuck in the "Metal Trial" stage 1 day before delivery, the Agent pings the responsible technician or manager on WhatsApp/Slack.

---

## 3. Engineering & Infrastructure Architecture

As an engineering manager, the priority is **Speed, Security, and Scalability**. 

### Stack & Infrastructure (GCP)
*   **Runtime:** **Bun** (for ultra-fast startup times and superior request handling).
*   **Web Framework:** **ElysiaJS** or **Hono** (Built for Bun, end-to-end type safety, native WebSocket support for streaming).
*   **Hosting:** Google Cloud Run (Serverless, scales to zero to save costs, massive horizontal scaling).
*   **Vector Database:** **GCP Cloud SQL (PostgreSQL + pgvector)** or **Pinecone**. Given GCP is the preferred infra, a dedicated Cloud SQL instance with pgvector ensures data stays within the GCP VPC, improving security and reducing latency.
*   **AI Models via ADK:** Utilize Google's Vertex AI (Gemini 2.0 Flash for speed/vision, Gemini 2.0 Pro for complex reasoning) and OpenAI APIs behind a unified internal gateway.

### Data Synchronization Layer (The "Brain Matrix")
The AI needs the main database's data to answer questions, but we cannot hammer the production MySQL database with complex analytic or RAG queries.
*   **Event-Driven Sync (Webhooks):** Whenever a Case (`Entry`), `Invoice`, or `Doctor` profile is updated in `app.dentnode.com`, a lightweight webhook is fired to `ai.dentnode.com`.
*   **Vectorization Pipeline:** `ai.dentnode.com` processes the webhook, converts the text/metadata into high-dimensional vector embeddings, and stores them in its dedicated Vector DB.
*   **Metadata Tagging:** All vectors will be strictly tagged with a `lab_id` to ensure multi-tenant security.

---

## 4. Security, Speed & Token Tracking

### Security (Multi-Tenancy & HIPAA-Lite)
*   **Hard Isolation:** Every vector DB query will strictly filter by `lab_id == request.lab_id`. A lab can *never* retrieve embeddings belonging to another lab.
*   **PII Scrubbing:** Before sending prompt data to an external LLM (OpenAI / Vertex), patient names and sensitive data will be optionally masked or replaced with synthetic IDs using an edge middleware.
*   **API Security:** Requests from `app.dentnode.com` to `ai.dentnode.com` will be secured via Internal JWTs or GCP IAM Service Accounts.

### Token Tracking & Billing Infrastructure
*   **Unified Ledger Table:** Create an `ai_token_ledgers` table in the AI service (or syncing back to main DB).
*   **Granular Metrics:** Every API call will log: `lab_id`, `feature_used` (e.g., WorkRx Scan vs. Chat), `model_used`, `input_tokens`, `output_tokens`.
*   **Hard Caps & Tiers:** Implement memory-cached (Redis/Bun in-memory) rate limiters. If a lab exceeds their monthly AI token quota (e.g., 50k tokens on a basic plan), the service gracefully degrades or blocks until upgrade.

---

## 5. Development Milestones

1.  **Phase 1: Foundation Server & Auth**
    *   Initialize Bun + Elysia project in `ai.dentnode.com`.
    *   Setup Prisma pointing to the new AI/Vector DB alongside the main MySQL DB (for read-only or webhook verification).
    *   Implement API Key / JWT Middleware and the Token Tracking engine.
2.  **Phase 2: The Vector Sync Engine**
    *   Establish webhook endpoints on `ai.dentnode.com`.
    *   Configure `app.dentnode.com` to securely broadcast changes.
    *   Write the embedding pipeline (Chunking -> Embedding -> Upsert to Vector DB).
3.  **Phase 3: Core AI Features Setup**
    *   Develop the Chat/RAG endpoint.
    *   Develop the WorkRx Vision scanner endpoint.
4.  **Phase 4: Agentic Automation**
    *   Deploy CRON-based or event-triggered agents that evaluate lab rules and execute automated actions.
5.  **Phase 5: GCP Deployment**
    *   Containerize via Dockerfile optimized for Bun.
    *   Deploy to GCP Cloud Run via CI/CD.
