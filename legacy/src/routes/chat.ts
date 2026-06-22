import { Elysia, t } from "elysia";
import { Stream } from "@elysiajs/stream";
import { searchSimilar } from "../services/embedder";
import { trackUsage } from "../services/tokenTracker";
import { scrubText } from "../services/scrubber";
import { generateGeminiText, GEMINI_CHAT_MODEL } from "../services/gemini";

const CHAT_MODEL = GEMINI_CHAT_MODEL;

type RetrievalSpec = {
  query: string;
  entityTypeFilter?: string;
  limit: number;
};

const CASE_QUERY_PATTERN = /\bcase|cases|delivery|deliver|dispatch|going|ready|patient|work\b/i;
const INVOICE_QUERY_PATTERN = /\binvoice|invoices|payment|payments|billing|bill|outstanding|balance\b/i;
const DOCTOR_QUERY_PATTERN = /\bdoctor|doctors|dentist|dentists|clinic|clinics\b/i;
const TEMPORAL_QUERY_PATTERN = /\btoday|tomorrow|yesterday\b/i;

const formatDate = (date: Date) => date.toISOString().slice(0, 10);

function getRelativeDateHints(message: string) {
  const lowerMessage = message.toLowerCase();
  const today = new Date();
  const hints: string[] = [];

  if (lowerMessage.includes("today")) {
    hints.push(`today = ${formatDate(today)}`);
  }

  if (lowerMessage.includes("tomorrow")) {
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);
    hints.push(`tomorrow = ${formatDate(tomorrow)}`);
  }

  if (lowerMessage.includes("yesterday")) {
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    hints.push(`yesterday = ${formatDate(yesterday)}`);
  }

  return hints;
}

function buildRetrievalSpecs(message: string): { specs: RetrievalSpec[]; relativeDateHints: string[] } {
  const specs: RetrievalSpec[] = [{ query: message, limit: 8 }];
  const relativeDateHints = getRelativeDateHints(message);
  const isInvoiceQuery = INVOICE_QUERY_PATTERN.test(message);
  const isDoctorQuery = DOCTOR_QUERY_PATTERN.test(message) && !isInvoiceQuery;
  const isCaseQuery = CASE_QUERY_PATTERN.test(message) && !isInvoiceQuery;

  if (isCaseQuery) {
    const caseSearchTerms = [message, "dental case"];

    if (TEMPORAL_QUERY_PATTERN.test(message)) {
      caseSearchTerms.push("expected delivery date", "dispatch", "going out");
      caseSearchTerms.push(...relativeDateHints.map((hint) => hint.split("=")[1]?.trim() ?? ""));
    }

    specs.unshift({
      query: caseSearchTerms.filter(Boolean).join(" "),
      entityTypeFilter: "case",
      limit: 12,
    });
  } else if (isInvoiceQuery) {
    specs.unshift({
      query: `${message} invoice payment due date outstanding amount`,
      entityTypeFilter: "invoice",
      limit: 10,
    });
  } else if (isDoctorQuery) {
    specs.unshift({
      query: `${message} doctor clinic profile`,
      entityTypeFilter: "doctor",
      limit: 10,
    });
  }

  return { specs, relativeDateHints };
}

export const chatRoutes = new Elysia({ prefix: "/chat" }).post(
  "/",
  async ({ body, request, set }) => {
    // Rely on middleware extracting labId. If internal tool skips middleware, we ensure it's provided.
    // The request should have been validated by authMiddleware, but we extract explicitly:
    const labId = body.labId;

    if (!labId) {
      set.status = 400;
      return { success: false, error: "labId is required for context isolation." };
    }

    // 1. Scrub User Query for PII
    const cleanMessage = scrubText(body.message);

    // 2. Semantic Search for context
    const { specs, relativeDateHints } = buildRetrievalSpecs(cleanMessage);
    const contextMap = new Map<string, Awaited<ReturnType<typeof searchSimilar>>[number]>();

    for (const spec of specs) {
      const results = await searchSimilar(labId, spec.query, spec.limit, spec.entityTypeFilter);

      for (const result of results) {
        const key = `${result.entityType}:${result.entityId}:${result.content}`;
        const existing = contextMap.get(key);
        if (!existing || result.similarity > existing.similarity) {
          contextMap.set(key, result);
        }
      }
    }

    const topContexts = Array.from(contextMap.values())
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, 12);

    const contextText = topContexts
      .map((ctx, i) => `[Source ${i + 1}] (${ctx.entityType} ID: ${ctx.entityId})\n${ctx.content}`)
      .join("\n\n");

    const todayIso = formatDate(new Date());
    const relativeDateContext =
      relativeDateHints.length > 0
        ? `Relative date references for this request: ${relativeDateHints.join("; ")}.`
        : "";

    const systemPrompt = `You are a highly capable AI assistant for a Dental Lab.
You exclusively answer questions regarding the lab operations, cases, doctors, and financial data based on the provided CONTEXT.
If the answer cannot be confidently deduced from the CONTEXT, explicitly state that you don't know rather than hallucinating.
Maintain a professional, helpful, and concise tone.
Today is ${todayIso}. ${relativeDateContext}
For questions about cases "going", "delivering", "dispatching", or due "today/tomorrow", interpret that using the case expected delivery date in the context.
When listing matching cases, prefer concise lines that include case ID, patient, doctor, and expected delivery date.

CURRENT CONTEXT (Database extracts for this lab):
${contextText.length > 0 ? contextText : "No relevant database records found for this query."}`;

    // 2. Prepare stream
    const completion = await generateGeminiText({
      model: CHAT_MODEL,
      systemInstruction: systemPrompt,
      prompt: body.message,
    });

    // 3. Stream back via Server-Sent Events (SSE)
    return new Stream(async (stream) => {
      try {
        if (completion.text) {
          stream.send(completion.text);
        }
      } catch (err) {
        console.error("Chat stream error:", err);
      } finally {
        stream.close();
        if (completion.inputTokens > 0) {
          // Log tokens asynchronously after the stream completes
          trackUsage(
            labId,
            "chat",
            CHAT_MODEL,
            completion.inputTokens,
            completion.outputTokens
          ).catch((e) =>
            console.error("Failed to track chat token usage:", e)
          );
        }
      }
    });
  },
  {
    body: t.Object({
      labId: t.String(),
      message: t.String(),
      conversationId: t.Optional(t.String()),
    }),
    detail: {
      tags: ["Chat"],
      summary: "Lab Assistant Chat (RAG) Stream",
      description:
        "Conversational AI endpoint powered by RAG over the lab's synced vector embeddings. Returns streaming responses via Server-Sent Events (SSE).",
    },
  }
);
