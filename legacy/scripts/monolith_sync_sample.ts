/**
 * MONOLITH INTEGRATION SAMPLE (Node.js/Bun)
 * 
 * This script demonstrates how to securely sync data from your main application
 * (app.dentnode.com) to the AI microservice.
 * 
 * 1. Prepare your data.
 * 2. Calculate the HMAC-SHA256 signature using your shared WEBHOOK_SECRET.
 * 3. POST to /webhooks/sync with the 'x-webhook-signature' header.
 */

import crypto from "crypto";

// ─── CONFIGURATION ───────────────────────────────────────────────────────────
const AI_SERVICE_URL = "https://dentnode-ai-prod-840065687221.asia-south1.run.app";
const WEBHOOK_SECRET = "L7kNmVbdCavy_XBbmfBRqPNWDZ9__1yN5HRrMCrzz3mbb8iBAkAsCxepaUBhL_d3"; // SHARED SECRET

// ─── DATA TO SYNC ────────────────────────────────────────────────────────────
const payload = {
  event: "case.updated",
  lab_id: "lab_67890",
  data: {
    id: "case_abc123",
    patientName: "John Doe",
    doctorName: "Dr. Arshad",
    product: "Zirconia Crown",
    shade: "A2",
    status: "In Progress",
    notes: "Patient requested high translucency on the incisal edge.",
  },
};

// ─── SIGNATURE CALCULATION ───────────────────────────────────────────────────
/**
 * Generates an HMAC-SHA256 signature for the payload string.
 */
function signPayload(body: string, secret: string): string {
  const hmac = crypto.createHmac("sha256", secret);
  hmac.update(body);
  return `sha256=${hmac.digest("hex")}`;
}

// ─── EXECUTION ───────────────────────────────────────────────────────────────
async function sync() {
  const body = JSON.stringify(payload);
  const signature = signPayload(body, WEBHOOK_SECRET);

  console.log(`📡 Syncing Case ${payload.data.id} to AI Service...`);

  try {
    const response = await fetch(`${AI_SERVICE_URL}/webhooks/sync`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-webhook-signature": signature,
      },
      body,
    });

    const result = await response.json();

    if (response.ok) {
      console.log("✅ Successfully synced:", result);
    } else {
      console.error("❌ Sync failed:", result);
    }
  } catch (error) {
    console.error("💥 Request error:", error);
  }
}

sync();
