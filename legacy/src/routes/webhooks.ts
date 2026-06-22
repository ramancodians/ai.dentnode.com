import { Elysia, t } from "elysia";
import { embedAndStore } from "../services/embedder";
import { trackUsage } from "../services/tokenTracker";
import { scrubEntityData } from "../services/scrubber";

// Supported event types from app.dentnode.com
type WebhookEvent = "case.updated" | "case.created" | "case.deleted"
  | "invoice.updated" | "invoice.created"
  | "doctor.updated" | "doctor.created";

const ENTITY_MAP: Record<string, string> = {
  "case.updated": "case",
  "case.created": "case",
  "case.deleted": "case",
  "invoice.updated": "invoice",
  "invoice.created": "invoice",
  "doctor.updated": "doctor",
  "doctor.created": "doctor",
};

// ─── HMAC signature validation ────────────────────────────────────────────────

async function verifyWebhookSignature(
  rawBody: string,
  signature: string | null
): Promise<boolean> {
  const webhookSecret = process.env.WEBHOOK_SECRET;
  if (!webhookSecret) {
    console.warn("⚠️  WEBHOOK_SECRET not set — skipping signature validation");
    return true; // In dev without a secret, allow through
  }
  if (!signature) return false;

  const encoder = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    encoder.encode(webhookSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify", "sign"]
  );

  const signedBuffer = await crypto.subtle.sign(
    "HMAC",
    keyMaterial,
    encoder.encode(rawBody)
  );

  const expectedHex = Array.from(new Uint8Array(signedBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  // Constant-time comparison
  const expected = `sha256=${expectedHex}`;
  if (expected.length !== signature.length) return false;

  let mismatch = 0;
  for (let i = 0; i < expected.length; i++) {
    mismatch |= expected.charCodeAt(i) ^ signature.charCodeAt(i);
  }
  return mismatch === 0;
}

// ─── Route ────────────────────────────────────────────────────────────────────

export const webhookRoutes = new Elysia({ prefix: "/webhooks" })
  .post(
    "/sync",
    async ({ body, request, set }) => {
      const rawBody = JSON.stringify(body);
      const signature = request.headers.get("x-webhook-signature");

      const isValid = await verifyWebhookSignature(rawBody, signature);
      if (!isValid) {
        set.status = 401;
        return { success: false, error: "Invalid webhook signature" };
      }

      const { event, lab_id, data } = body as {
        event: WebhookEvent;
        lab_id: string;
        data: Record<string, unknown>;
      };

      if (!lab_id || !event || !data) {
        set.status = 400;
        return { success: false, error: "Missing required fields: event, lab_id, data" };
      }

      const entityType = ENTITY_MAP[event];

      // Handle deletions — remove embeddings instead of upserting
      if (event.endsWith(".deleted")) {
        const { prisma } = await import("../db/client");
        const entityId = String(data.id ?? "");
        await prisma.vectorEmbedding.deleteMany({ where: { labId: lab_id, entityType, entityId } });
        return { success: true, action: "deleted", entityType, entityId };
      }

      if (!entityType) {
        set.status = 422;
        return { success: false, error: `Unknown event type: ${event}` };
      }

      const entityId = String(data.id ?? "");
      if (!entityId) {
        set.status = 422;
        return { success: false, error: "data.id is required" };
      }

      // 1. Scrub PII if enabled
      const scrubbedData = scrubEntityData(data);

      // 2. Embed and store (asynchronously processing in the background)
      const startTime = Date.now();
      const result = await embedAndStore(lab_id, entityType, entityId, scrubbedData);



      // Track token usage for embedding
      if (result.inputTokens > 0) {
        await trackUsage(
          lab_id,
          "webhook_sync",
          process.env.EMBEDDING_MODEL ?? "text-embedding-3-small",
          result.inputTokens,
          0
        );
      }

      return {
        success: true,
        event,
        lab_id,
        entityType,
        entityId,
        chunksProcessed: result.chunksProcessed,
        inputTokens: result.inputTokens,
        processingMs: Date.now() - startTime,
      };
    },
    {
      body: t.Object({
        event: t.String(),
        lab_id: t.String(),
        data: t.Record(t.String(), t.Unknown()),
      }),
      detail: {
        tags: ["Webhooks"],
        summary: "Sync entity data from app.dentnode.com",
        description:
          "Receives entity change events from the main app, validates HMAC signature, and upserts vector embeddings scoped to the lab.",
      },
    }
  );
