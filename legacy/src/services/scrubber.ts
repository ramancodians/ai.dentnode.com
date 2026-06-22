/**
 * PII Scrubber Service
 * Responsible for masking or anonymizing sensitive patient data
 * (Patient Names, Phone Numbers, Emails) before they are sent to external LLMs.
 */

export interface ScrubbingOptions {
  strategy: "mask" | "anonymize";
  enabled: boolean;
}

const DEFAULT_OPTIONS: ScrubbingOptions = {
  strategy: "anonymize",
  enabled: process.env.ENABLE_PII_SCRUBBING === "true",
};

/**
 * Regex patterns for common PII
 */
const PATTERNS = {
  email: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
  phone: /(\+?\d{1,4}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g,
  // Note: Patient Names are harder to catch with generic Regex without false positives.
  // We prioritize structured field scrubbing (e.g. "patient_name" in raw data).
};

/**
 * Scrubs a string of text using patterns
 */
export function scrubText(text: string, options = DEFAULT_OPTIONS): string {
  if (!options.enabled) return text;

  let scrubbed = text;
  const replacement = options.strategy === "mask" ? "[SCRUBBED]" : "ANON_USER";

  scrubbed = scrubbed.replace(PATTERNS.email, options.strategy === "mask" ? "[EMAIL]" : "anon@example.com");
  scrubbed = scrubbed.replace(PATTERNS.phone, options.strategy === "mask" ? "[PHONE]" : "555-0199");

  return scrubbed;
}

/**
 * Recursively scrubs an object, specifically looking for known patient fields
 */
export function scrubEntityData(data: any, options = DEFAULT_OPTIONS): any {
  if (!options.enabled || !data || typeof data !== "object") return data;

  const sensitiveFields = ["patient_name", "patientName", "patient_phone", "patientPhone", "patient_email", "patientEmail", "address"];
  
  // Clone to avoid mutating original
  const result = Array.isArray(data) ? [...data] : { ...data };

  for (const key in result) {
    const value = result[key];

    if (sensitiveFields.includes(key) && typeof value === "string") {
      if (options.strategy === "mask") {
        result[key] = `[${key.toUpperCase()}_SCRUBBED]`;
      } else {
        // Simple synthetic anonymization
        result[key] = `Patient_${Math.abs(hashString(value)).toString(16).substring(0, 4)}`;
      }
    } else if (typeof value === "object" && value !== null) {
      result[key] = scrubEntityData(value, options);
    } else if (typeof value === "string") {
      result[key] = scrubText(value, options);
    }
  }

  return result;
}

/**
 * Deterministic hash for anonymization consistency
 */
function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
}
