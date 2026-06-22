import prisma from "../db/client";

// ─── In-memory quota cache (reset lazily on DB-side reset) ───────────────────
// Keyed by labId → { tokensUsed, cap, cachedAt }
const quotaCache = new Map<
  string,
  { tokensUsed: number; cap: number; cachedAt: number; resetAt: Date }
>();
const CACHE_TTL_MS = 60_000; // 1 minute

// ─── Cost estimation (approximate USD) ───────────────────────────────────────
const COST_PER_1K: Record<string, { input: number; output: number }> = {
  "gpt-4o": { input: 0.005, output: 0.015 },
  "gpt-4o-mini": { input: 0.00015, output: 0.0006 },
  "text-embedding-3-small": { input: 0.00002, output: 0 },
  "gemini-2.0-flash": { input: 0.00035, output: 0.00105 },
  "gemini-2.0-pro": { input: 0.00125, output: 0.005 },
  "embedding-001": { input: 0.000025, output: 0 },
  "gemini-2.5-flash": { input: 0.0003, output: 0.0025 },
  "text-embedding-004": { input: 0.00002, output: 0 },
};

function estimateCost(model: string, inputTokens: number, outputTokens: number): number {
  const rates = COST_PER_1K[model] ?? { input: 0, output: 0 };
  return (inputTokens / 1000) * rates.input + (outputTokens / 1000) * rates.output;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Record AI usage for a lab and update its running monthly total.
 */
export async function trackUsage(
  labId: string,
  featureUsed: string,
  modelUsed: string,
  inputTokens: number,
  outputTokens: number
): Promise<void> {
  const totalTokens = inputTokens + outputTokens;
  const costUsd = estimateCost(modelUsed, inputTokens, outputTokens);

  await Promise.all([
    // Write to ledger
    prisma.aiTokenLedger.create({
      data: { labId, featureUsed, modelUsed, inputTokens, outputTokens, costUsd },
    }),
    // Upsert quota counter (idempotent on first use)
    prisma.labQuota.upsert({
      where: { labId },
      create: {
        labId,
        tokensUsedThisMonth: totalTokens,
        monthlyTokenCap: 50_000,
        tier: "basic",
        resetAt: nextMonthReset(),
      },
      update: {
        tokensUsedThisMonth: { increment: totalTokens },
        updatedAt: new Date(),
      },
    }),
  ]);

  // Invalidate cache for this lab so next checkQuota re-reads truth
  quotaCache.delete(labId);
}

/**
 * Check if a lab is within its monthly quota.
 * Uses an in-memory cache (1 min TTL) to avoid hammering the DB on every request.
 */
export async function checkQuota(
  labId: string
): Promise<{ allowed: boolean; remaining: number; cap: number }> {
  const now = Date.now();
  const cached = quotaCache.get(labId);

  if (cached && now - cached.cachedAt < CACHE_TTL_MS) {
    const remaining = Math.max(0, cached.cap - cached.tokensUsed);
    return { allowed: remaining > 0, remaining, cap: cached.cap };
  }

  // Fetch from DB (upsert so first-time labs get a quota row)
  const quota = await prisma.labQuota.upsert({
    where: { labId },
    create: {
      labId,
      tokensUsedThisMonth: 0,
      monthlyTokenCap: 50_000,
      tier: "basic",
      resetAt: nextMonthReset(),
    },
    update: {},
  });

  // Auto-reset if we've passed the reset date
  if (new Date() >= quota.resetAt) {
    await prisma.labQuota.update({
      where: { labId },
      data: { tokensUsedThisMonth: 0, resetAt: nextMonthReset() },
    });
    quota.tokensUsedThisMonth = 0;
  }

  quotaCache.set(labId, {
    tokensUsed: quota.tokensUsedThisMonth,
    cap: quota.monthlyTokenCap,
    cachedAt: now,
    resetAt: quota.resetAt,
  });

  const remaining = Math.max(0, quota.monthlyTokenCap - quota.tokensUsedThisMonth);
  return { allowed: remaining > 0, remaining, cap: quota.monthlyTokenCap };
}

/**
 * Get usage summary for a lab (last 30 days + current month total).
 */
export async function getLabUsageSummary(labId: string) {
  const [quota, ledgerSummary] = await Promise.all([
    prisma.labQuota.findUnique({ where: { labId } }),
    prisma.aiTokenLedger.groupBy({
      by: ["featureUsed", "modelUsed"],
      where: { labId, createdAt: { gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) } },
      _sum: { inputTokens: true, outputTokens: true, costUsd: true },
      _count: { id: true },
    }),
  ]);

  return { quota, breakdown: ledgerSummary };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function nextMonthReset(): Date {
  const d = new Date();
  d.setUTCMonth(d.getUTCMonth() + 1, 1);
  d.setUTCHours(0, 0, 0, 0);
  return d;
}
