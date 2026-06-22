import { generateGeminiText, GEMINI_AGENT_MODEL } from "./gemini";

const EVALUATOR_MODEL = GEMINI_AGENT_MODEL;

export interface AgentRule {
  id: string;
  description: string;
}

export interface EvaluationResult {
  requiresAction: boolean;
  actionType: "ALERT_CLINIC" | "ALERT_MANAGER" | "NONE";
  reason: string;
  draftMessage: string | null;
}

/**
 * AI Evaluator: Looks at a specific case and comparing it against
 * the lab's rules to determine if an intervention is required.
 */
export async function evaluateCaseRules(
  caseData: Record<string, any>,
  labRules: AgentRule[]
): Promise<EvaluationResult & { inputTokens: number; outputTokens: number }> {
  if (labRules.length === 0) {
    return {
      requiresAction: false,
      actionType: "NONE",
      reason: "No rules defined.",
      draftMessage: null,
      inputTokens: 0,
      outputTokens: 0,
    };
  }

  const prompt = `You are a strict, intelligent Quality Control Agent for a Dental Lab.
You must review the provided CASE DATA against the established LAB RULES.
Analyze if the case currently violates or triggers any of the rules requiring intervention.

LAB RULES:
${labRules.map((r, i) => `${i + 1}. ${r.description}`).join("\n")}

CASE DATA:
${JSON.stringify(caseData, null, 2)}

Provide your assessment in strict JSON format. Do not use markdown blocks, just the raw JSON:
{
  "requiresAction": boolean, // true if a rule is triggered
  "actionType": "string", // only use: "ALERT_CLINIC", "ALERT_MANAGER", or "NONE"
  "reason": "string", // short explanation of why the rule was triggered
  "draftMessage": "string" // A polite drafted message to send (Whatsapp/Email) resolving the issue. Use null if no action needed.
}`;

  const response = await generateGeminiText({
    model: EVALUATOR_MODEL,
    prompt,
    responseMimeType: "application/json",
  });

  const content = response.text || "{}";
  const parsed = JSON.parse(content);

  return {
    requiresAction: !!parsed.requiresAction,
    actionType: parsed.actionType || "NONE",
    reason: parsed.reason || "",
    draftMessage: parsed.draftMessage || null,
    inputTokens: response.inputTokens || 0,
    outputTokens: response.outputTokens || 0,
  };
}
