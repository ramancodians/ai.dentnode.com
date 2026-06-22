type GeminiGenerationInput = {
  prompt: string;
  systemInstruction?: string;
  model?: string;
  temperature?: number;
  responseMimeType?: string;
};

type EmbeddingTaskType = "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
export const GEMINI_CHAT_MODEL =
  process.env.GEMINI_CHAT_MODEL || "gemini-2.5-flash";
export const GEMINI_AGENT_MODEL =
  process.env.GEMINI_AGENT_MODEL || GEMINI_CHAT_MODEL;
export const GEMINI_VISION_MODEL =
  process.env.GEMINI_VISION_MODEL || GEMINI_CHAT_MODEL;
export const GEMINI_EMBEDDING_MODEL =
  process.env.EMBEDDING_MODEL || "text-embedding-004";

const estimateTokens = (text: string) => Math.max(1, Math.ceil(text.length / 4));

const extractGeminiText = (payload: any) =>
  (payload?.candidates || [])
    .flatMap((candidate: any) => candidate?.content?.parts || [])
    .map((part: any) => part?.text || "")
    .join("")
    .trim();

const buildError = async (response: Response) => {
  const errorText = await response.text();
  return new Error(`Gemini request failed (${response.status}): ${errorText}`);
};

const fetchGemini = async (path: string, body: Record<string, unknown>) => {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/${path}?key=${GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    }
  );

  if (!response.ok) {
    throw await buildError(response);
  }

  return response.json();
};

export async function generateGeminiText(
  input: GeminiGenerationInput
): Promise<{ text: string; inputTokens: number; outputTokens: number; model: string }> {
  const model = input.model || GEMINI_CHAT_MODEL;
  const payload = await fetchGemini(`models/${model}:generateContent`, {
    contents: [
      {
        role: "user",
        parts: [{ text: input.prompt }],
      },
    ],
    ...(input.systemInstruction
      ? {
          systemInstruction: {
            parts: [{ text: input.systemInstruction }],
          },
        }
      : {}),
    generationConfig: {
      temperature: input.temperature ?? 0.2,
      ...(input.responseMimeType
        ? { responseMimeType: input.responseMimeType }
        : {}),
    },
  });

  const text = extractGeminiText(payload);

  if (!text) {
    throw new Error("Gemini API returned an empty response.");
  }

  return {
    text,
    inputTokens:
      Number(payload?.usageMetadata?.promptTokenCount || 0) ||
      estimateTokens(input.prompt),
    outputTokens:
      Number(payload?.usageMetadata?.candidatesTokenCount || 0) ||
      estimateTokens(text),
    model,
  };
}

export async function embedTextsWithGemini(
  texts: string[],
  taskType: EmbeddingTaskType
): Promise<{ embeddings: number[][]; inputTokens: number }> {
  const model = GEMINI_EMBEDDING_MODEL;
  let estimatedInputTokens = 0;

  const embeddings = await Promise.all(
    texts.map(async (text) => {
      estimatedInputTokens += estimateTokens(text);

      const payload = await fetchGemini(`models/${model}:embedContent`, {
        model: `models/${model}`,
        content: {
          parts: [{ text }],
        },
        taskType,
      });

      const values = payload?.embedding?.values;
      if (!Array.isArray(values) || values.length === 0) {
        throw new Error("Gemini embedding API returned no vector values.");
      }

      return values.map((value: number) => Number(value));
    })
  );

  return {
    embeddings,
    inputTokens: estimatedInputTokens,
  };
}