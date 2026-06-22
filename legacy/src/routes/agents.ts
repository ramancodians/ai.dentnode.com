import { Elysia, t } from "elysia";
import { evaluateCaseRules, type AgentRule } from "../services/agent";
import { trackUsage } from "../services/tokenTracker";
import { GEMINI_AGENT_MODEL } from "../services/gemini";

export const agentRoutes = new Elysia({ prefix: "/agents" }).post(
  "/evaluate-case",
  async ({ body, request, set }) => {
    const labId = request.headers.get("x-lab-id");
    if (!labId) {
      set.status = 400;
      return { success: false, error: "Missing x-lab-id header" };
    }

    try {
      const startTime = Date.now();
      
      const result = await evaluateCaseRules(body.caseData, body.labRules);

      if (result.inputTokens > 0) {
        // Track usage asynchronously
        trackUsage(
          labId,
          "agent_evaluation",
          GEMINI_AGENT_MODEL,
          result.inputTokens,
          result.outputTokens
        ).catch(err => console.error("Token tracking failed:", err));
      }

      return {
        success: true,
        processingMs: Date.now() - startTime,
        evaluation: {
          requiresAction: result.requiresAction,
          actionType: result.actionType,
          reason: result.reason,
          draftMessage: result.draftMessage,
        },
      };

    } catch (err: any) {
      console.error("Agent Evaluation Error:", err);
      set.status = 500;
      return { success: false, error: "Failed to evaluate case rules" };
    }
  },
  {
    body: t.Object({
      caseData: t.Record(t.String(), t.Any()),
      labRules: t.Array(
        t.Object({
          id: t.String(),
          description: t.String(),
        })
      ),
    }),
    detail: {
      tags: ["Agent Automation"],
      summary: "Evaluate Case against Lab Rules",
      description: "Applies 'Background Worker' logic to a case payload. Evaluates if the case triggers any custom rules (e.g. 'Stuck in Metal Trial') and drafts a notification message.",
    },
  }
);
