import { Elysia } from "elysia";
import { checkQuota } from "../services/tokenTracker";

// Routes that do NOT require any authentication
const PUBLIC_ROUTES = new Set(["/", "/health", "/docs", "/docs/json"]);

export const authMiddleware = new Elysia()
  .onBeforeHandle(async ({ request, set, path }) => {
    // Allow public routes through without any checks
    if (PUBLIC_ROUTES.has(path)) return;

    // ── Internal key validation ────────────────────────────────────────────
    const internalKey = process.env.AI_SERVICE_INTERNAL_KEY;
    const providedKey = request.headers.get("x-ai-internal-key");

    if (!internalKey) {
      console.warn(
        "⚠️  AI_SERVICE_INTERNAL_KEY is not set — all non-public requests will be rejected."
      );
      set.status = 503;
      return { success: false, error: "Service misconfigured: internal key not set" };
    }

    if (!providedKey || providedKey !== internalKey) {
      set.status = 401;
      return { success: false, error: "Unauthorized: invalid or missing x-ai-internal-key" };
    }

    // ── Quota enforcement (skip for admin & webhook routes) ───────────────
    const labId = request.headers.get("x-lab-id");
    const isAdminRoute = path.startsWith("/admin");
    const isWebhookRoute = path.startsWith("/webhooks");

    if (labId && !isAdminRoute && !isWebhookRoute) {
      const quota = await checkQuota(labId);
      if (!quota.allowed) {
        set.status = 429;
        return {
          success: false,
          error: "Monthly AI token quota exceeded",
          remaining: 0,
          cap: quota.cap,
          hint: "Upgrade your plan or wait for the monthly reset.",
        };
      }
    }
  })
  .derive(({ request }) => ({
    labId: request.headers.get("x-lab-id") ?? null,
    isAuthenticated: true,
  }));
