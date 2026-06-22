import { Elysia, t } from "elysia";
import { getLabUsageSummary } from "../services/tokenTracker";
import prisma from "../db/client";

export const adminRoutes = new Elysia({ prefix: "/admin" })
  // ─── Lab quota & usage summary ─────────────────────────────────────────────
  .get(
    "/quota/:labId",
    async ({ params: { labId }, set }) => {
      const summary = await getLabUsageSummary(labId);
      if (!summary.quota) {
        set.status = 404;
        return { success: false, error: `No quota record found for lab: ${labId}` };
      }
      return { success: true, labId, ...summary };
    },
    {
      params: t.Object({ labId: t.String() }),
      detail: {
        tags: ["Admin"],
        summary: "Get token usage & quota for a lab",
      },
    }
  )

  // ─── All labs overview ─────────────────────────────────────────────────────
  .get(
    "/quotas",
    async () => {
      const quotas = await prisma.labQuota.findMany({
        orderBy: { tokensUsedThisMonth: "desc" },
      });
      return { success: true, count: quotas.length, quotas };
    },
    {
      detail: {
        tags: ["Admin"],
        summary: "List all lab quotas (sorted by usage desc)",
      },
    }
  )

  // ─── Update a lab's quota cap / tier ──────────────────────────────────────
  .patch(
    "/quota/:labId",
    async ({ params: { labId }, body }) => {
      const quota = await prisma.labQuota.upsert({
        where: { labId },
        create: {
          labId,
          tier: body.tier ?? "basic",
          monthlyTokenCap: body.monthlyTokenCap ?? 50_000,
          tokensUsedThisMonth: 0,
        },
        update: {
          ...(body.tier ? { tier: body.tier } : {}),
          ...(body.monthlyTokenCap ? { monthlyTokenCap: body.monthlyTokenCap } : {}),
        },
      });
      return { success: true, quota };
    },
    {
      params: t.Object({ labId: t.String() }),
      body: t.Object({
        tier: t.Optional(t.String()),
        monthlyTokenCap: t.Optional(t.Number()),
      }),
      detail: {
        tags: ["Admin"],
        summary: "Update a lab's quota cap or tier",
      },
    }
  )

  // ─── Recent ledger entries ─────────────────────────────────────────────────
  .get(
    "/ledger/:labId",
    async ({ params: { labId }, query }) => {
      const limit = Math.min(Number(query.limit ?? 50), 200);
      const entries = await prisma.aiTokenLedger.findMany({
        where: { labId },
        orderBy: { createdAt: "desc" },
        take: limit,
      });
      return { success: true, labId, count: entries.length, entries };
    },
    {
      params: t.Object({ labId: t.String() }),
      query: t.Object({ limit: t.Optional(t.String()) }),
      detail: {
        tags: ["Admin"],
        summary: "Get recent token ledger entries for a lab",
      },
    }
  );
