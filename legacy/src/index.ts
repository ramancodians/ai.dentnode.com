import { Elysia } from "elysia";
import { cors } from "@elysiajs/cors";
import { swagger } from "@elysiajs/swagger";
import { authMiddleware } from "./middleware/auth";
import { webhookRoutes } from "./routes/webhooks";
import { adminRoutes } from "./routes/admin";
import { chatRoutes } from "./routes/chat";
import { scanRoutes } from "./routes/scan";
import { agentRoutes } from "./routes/agents";

const PORT = Number(process.env.PORT ?? 8000);
const NODE_ENV = process.env.NODE_ENV ?? "development";

// We bump the maxRequestBodySize in listen() to handle up to 10MB WorkRx images (default is 1mb).
const app = new Elysia()

  // ── Global plugins ──────────────────────────────────────────────────────────
  .use(
    cors({
      origin: [
        "https://app.dentnode.com",
        "https://clinic.dentnode.com",
        // Allow localhost in dev
        ...(NODE_ENV === "development" ? [/localhost/] : []),
      ],
      allowedHeaders: ["Content-Type", "x-ai-internal-key", "x-lab-id", "x-webhook-signature"],
      methods: ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
      credentials: true,
    })
  )
  .use(
    swagger({
      documentation: {
        info: {
          title: "DentNode AI Service",
          version: "0.1.0",
          description:
            "The intelligent Brain for the DentNode ecosystem. Handles AI features, vector sync, token tracking, and lab automation.",
        },
        tags: [
          { name: "Health", description: "Service health & status" },
          { name: "Webhooks", description: "Entity sync from app.dentnode.com" },
          { name: "Chat", description: "Lab assistant RAG chat (Phase 3)" },
          { name: "Admin", description: "Quota management & usage analytics" },
        ],
        components: {
          securitySchemes: {
            InternalKey: {
              type: "apiKey",
              in: "header",
              name: "x-ai-internal-key",
            },
          },
        },
      },
    })
  )
  // ── Auth (public routes bypass inside middleware) ────────────────────────────
  .use(authMiddleware)
  // ── Public routes ───────────────────────────────────────────────────────────
  .get(
    "/",
    () => ({
      service: "DentNode AI Brain",
      status: "online",
      docs: "/docs",
    }),
    {
      detail: { tags: ["Health"], summary: "Service root" },
    }
  )
  .get(
    "/health",
    () => ({
      success: true,
      service: "ai.dentnode.com",
      status: "healthy",
      environment: NODE_ENV,
      timestamp: new Date().toISOString(),
    }),
    {
      detail: { tags: ["Health"], summary: "Health check (no auth required)" },
    }
  )
  // ── Protected route groups ───────────────────────────────────────────────────
  .use(webhookRoutes)
  .use(adminRoutes)
  .use(chatRoutes)
  .use(scanRoutes)
  .use(agentRoutes)
  .listen({ port: PORT, maxRequestBodySize: 10 * 1024 * 1024 });


console.log(
  `🦷 DentNode AI Brain is live → http://${app.server?.hostname}:${app.server?.port}`
);
console.log(`📖 Swagger docs    → http://${app.server?.hostname}:${app.server?.port}/docs`);

