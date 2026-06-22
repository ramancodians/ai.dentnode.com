import { Elysia, t } from "elysia";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { trackUsage } from "../services/tokenTracker";
import { GEMINI_VISION_MODEL } from "../services/gemini";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");
const VISION_MODEL = GEMINI_VISION_MODEL;

// Helper: Convert File/Blob to base64 for Gemini
async function fileToGenerativePart(file: File) {
  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  return {
    inlineData: {
      data: buffer.toString("base64"),
      mimeType: file.type,
    },
  };
}

export const scanRoutes = new Elysia({ prefix: "/scan" }).post(
  "/work-rx",
  async ({ body, request, set }) => {
    const labId = request.headers.get("x-lab-id");
    if (!labId) {
      set.status = 400;
      return { success: false, error: "Missing x-lab-id header" };
    }

    const file = body.image;
    if (!file) {
      set.status = 400;
      return { success: false, error: "Image file is required" };
    }

    try {
      const model = genAI.getGenerativeModel({ model: VISION_MODEL });

      const prompt = `You are a highly accurate dental document parser.
I am providing an image of a handwritten or printed dental Work Prescription (WorkRx) / Lab docket.
Extract the data and return ONLY a strict JSON object with the following schema:
{
  "patientName": "Extracted patient name or null",
  "doctorName": "Extracted dentist/clinic name or null",
  "product": "What type of product/restoration is requested (e.g., Zirconia Crown, PFM, Emax)? Return null if unclear",
  "shade": "What is the requested tooth shade (e.g., A2, B1, 3M2)? Return null if not specified",
  "teeth": ["Array of tooth numbers/notation (e.g., 'UR6', '16', 'LR7'). Empty array if none"],
  "dueDate": "Requested delivery/due date in YYYY-MM-DD format if possible or raw string, null if missing",
  "notes": "Any other specific instructions provided by the doctor. Leave empty string if none."
}`;

      const imagePart = await fileToGenerativePart(file as File);
      
      const startTime = Date.now();
      const result = await model.generateContent([prompt, imagePart]);
      const response = await result.response;
      let text = response.text();

      // Clean markdown block if Gemini wraps it
      text = text.replace(/^```json/g, "").replace(/```$/g, "").trim();
      const parsedData = JSON.parse(text);

      // Track rough usage (Gemini response doesn't provide precise tokens inline via simple API easily,
      // but we can estimate: ~260 for prompt, ~260 per image (reduced), ~100 completion)
      // Gemini Flash cost is exceptionally low, but logging is required for limits.
      // We log a flat ~1000 input tokens per image scan.
      await trackUsage(labId, "scan", VISION_MODEL, 1000, 150)
        .catch(console.error);

      return {
        success: true,
        processingMs: Date.now() - startTime,
        data: parsedData,
      };

    } catch (err: any) {
      console.error("Vision Scan Error:", err);
      set.status = 500;
      return { success: false, error: "Failed to parse WorkRx image: " + err.message };
    }
  },
  {
    // Important: Increase body limit here to 10MB to handle high-res photos
    body: t.Object({
      image: t.File(),
    }),
    type: "multipart/form-data",
    detail: {
      tags: ["Vision"],
      summary: "Scan WorkRx (Prescription) Image",
      description: "Uploads a photo of a dental prescription. Uses Gemini 2.5 Flash to extract product, shade, teeth, and notes into structured JSON.",
    },
  }
);
