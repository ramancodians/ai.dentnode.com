import prisma from "../db/client";
import { embedTextsWithGemini, GEMINI_EMBEDDING_MODEL } from "./gemini";

const EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL;
const MAX_CHUNK_TOKENS = 500; // ~500 tokens ≈ ~375 words — safe for embedding models
type LooseRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is LooseRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const toNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
};

const toDateString = (value: unknown): string | undefined => {
  if (!value) {
    return undefined;
  }

  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value.toISOString().slice(0, 10);
  }

  if (typeof value === "number") {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString().slice(0, 10);
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return undefined;
    }

    const parsed = new Date(trimmed);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().slice(0, 10);
    }

    return trimmed;
  }

  return undefined;
};

const splitValues = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.flatMap((entry) => splitValues(entry));
  }

  const normalized = toNonEmptyString(value);
  if (!normalized) {
    return [];
  }

  return normalized
    .split(/\s*,\s*/)
    .map((entry) => entry.trim())
    .filter(Boolean);
};

const uniqueValues = (values: unknown[]): string[] =>
  Array.from(
    new Set(
      values
        .flatMap((value) => splitValues(value))
        .map((value) => value.trim())
        .filter((value) => value && value.toUpperCase() !== "NA")
    )
  );

const buildName = (first?: unknown, last?: unknown) => {
  const fullName = [toNonEmptyString(first), toNonEmptyString(last)].filter(Boolean).join(" ").trim();
  return fullName || undefined;
};

const getNestedName = (value: unknown) => {
  if (!isRecord(value)) {
    return undefined;
  }

  return (
    buildName(value.first_name, value.last_name) ??
    buildName(value.firstName, value.lastName) ??
    toNonEmptyString(value.name)
  );
};

const mergeNotes = (...values: unknown[]) => {
  const notes = uniqueValues(values);
  return notes.length > 0 ? notes.join(" | ") : undefined;
};

function normalizeEntityData(entityType: string, data: LooseRecord): LooseRecord {
  switch (entityType) {
    case "case": {
      const workItems = Array.isArray(data.workItems)
        ? data.workItems.filter(isRecord)
        : [];

      const productNames = uniqueValues([
        data.product,
        data.products,
        data.productNames,
        ...workItems.map((workItem) => workItem.productName ?? workItem.product),
      ]);

      const shades = uniqueValues([
        data.shade,
        data.shades,
        ...workItems.map((workItem) => workItem.shade),
      ]);

      const departments = uniqueValues([
        data.department,
        data.departments,
        data.departmentNames,
        ...workItems.map((workItem) => workItem.departmentName ?? workItem.department),
      ]);

      const doctorName =
        toNonEmptyString(data.doctorName) ??
        buildName(data.doctor_first, data.doctor_last) ??
        getNestedName(data.doctor);

      const patientName =
        toNonEmptyString(data.patientName) ??
        buildName(data.patient_first, data.patient_last) ??
        getNestedName(data.patient);

      return {
        id:
          toNonEmptyString(data.case_custom_id) ??
          toNonEmptyString(data.caseCustomId) ??
          toNonEmptyString(data.id) ??
          "Unknown",
        status: toNonEmptyString(data.status),
        doctorName,
        patientName,
        product: productNames.join(", ") || undefined,
        shade: shades.join(", ") || undefined,
        department: departments.join(", ") || undefined,
        dueDate:
          toDateString(data.dueDate) ??
          toDateString(data.epxected_delivery_date) ??
          toDateString(data.expected_delivery_date) ??
          toDateString(data.delivery_date) ??
          toDateString(data.ready_date),
        expectedDeliverySlot:
          toNonEmptyString(data.expectedDeliverySlot) ??
          toNonEmptyString(data.expected_delivery_slot),
        deliveryType:
          toNonEmptyString(data.deliveryType) ??
          toNonEmptyString(data.delivery_type),
        deliveryNotes:
          toNonEmptyString(data.deliveryNotes) ??
          toNonEmptyString(data.delivery_notes),
        notes: mergeNotes(data.notes, data.deliveryNotes, data.delivery_notes),
        createdAt:
          toDateString(data.createdAt) ??
          toDateString(data.created_at),
      };
    }

    case "invoice": {
      return {
        id: toNonEmptyString(data.id) ?? "Unknown",
        doctorName:
          toNonEmptyString(data.doctorName) ??
          buildName(data.doctor_first, data.doctor_last) ??
          getNestedName(data.doctor),
        amount: data.amount ?? data.total_amount ?? data.bill_amount,
        status: toNonEmptyString(data.status),
        dueDate: toDateString(data.dueDate) ?? toDateString(data.due_date),
        notes: mergeNotes(data.notes),
      };
    }

    case "doctor": {
      return {
        name:
          toNonEmptyString(data.name) ??
          buildName(data.first_name, data.last_name) ??
          "Unknown",
        clinicName: toNonEmptyString(data.clinicName) ?? toNonEmptyString(data.clinic_name),
        specialty: toNonEmptyString(data.specialty),
        email: toNonEmptyString(data.email),
        phone: toNonEmptyString(data.phone),
        address: toNonEmptyString(data.address) ?? toNonEmptyString(data.full_address),
        preferences: toNonEmptyString(data.preferences),
      };
    }

    default:
      return data;
  }
}

// ─── Text Chunking ────────────────────────────────────────────────────────────

/**
 * Splits a text into chunks of approximately maxWords words, respecting sentence boundaries.
 */
export function chunkText(text: string, maxWords = MAX_CHUNK_TOKENS): string[] {
  if (!text || text.trim().length === 0) return [];

  // Split on sentence boundaries
  const sentences = text.match(/[^.!?]+[.!?]*/g) ?? [text];
  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let wordCount = 0;

  for (const sentence of sentences) {
    const words = sentence.trim().split(/\s+/).filter(Boolean);
    if (wordCount + words.length > maxWords && currentChunk.length > 0) {
      chunks.push(currentChunk.join(" ").trim());
      currentChunk = [];
      wordCount = 0;
    }
    currentChunk.push(sentence.trim());
    wordCount += words.length;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(" ").trim());
  }

  return chunks.filter((c) => c.length > 0);
}

// ─── Build Embeddable Text ────────────────────────────────────────────────────

/**
 * Converts a raw entity payload into a human-readable text for embedding.
 * Add more entity types as needed.
 */
export function buildEmbeddableText(entityType: string, data: Record<string, unknown>): string {
  const normalizedData = normalizeEntityData(entityType, data);
  const parts: string[] = [];

  switch (entityType) {
    case "case":
      parts.push(`Dental Case #${normalizedData.id ?? "Unknown"}`);
      if (normalizedData.status) parts.push(`Current Status: ${normalizedData.status}`);
      if (normalizedData.doctorName) parts.push(`Dentist/Doctor: ${normalizedData.doctorName}`);
      if (normalizedData.patientName) parts.push(`Patient Name: ${normalizedData.patientName}`);
      if (normalizedData.product) parts.push(`Prosthetic Products: ${normalizedData.product}`);
      if (normalizedData.shade) parts.push(`Tooth Shades: ${normalizedData.shade}`);
      if (normalizedData.dueDate) {
        parts.push(`Expected Delivery Date: ${normalizedData.dueDate}`);
        parts.push(`Going Out / Dispatch Date: ${normalizedData.dueDate}`);
      }
      if (normalizedData.expectedDeliverySlot) {
        parts.push(`Expected Delivery Slot: ${normalizedData.expectedDeliverySlot}`);
      }
      if (normalizedData.deliveryType) parts.push(`Delivery Type: ${normalizedData.deliveryType}`);
      if (normalizedData.department) parts.push(`Processing Departments: ${normalizedData.department}`);
      if (normalizedData.deliveryNotes) parts.push(`Delivery Notes: ${normalizedData.deliveryNotes}`);
      if (normalizedData.notes) parts.push(`Clinical Notes: ${normalizedData.notes}`);
      if (normalizedData.createdAt) parts.push(`Case Created On: ${normalizedData.createdAt}`);
      break;

    case "invoice":
      parts.push(`Invoice #${normalizedData.id ?? "Unknown"}`);
      if (normalizedData.doctorName) parts.push(`Billed to Doctor: ${normalizedData.doctorName}`);
      if (normalizedData.amount) parts.push(`Total Amount: ${normalizedData.amount}`);
      if (normalizedData.status) parts.push(`Payment Status: ${normalizedData.status}`);
      if (normalizedData.dueDate) parts.push(`Payment Due Date: ${normalizedData.dueDate}`);
      if (normalizedData.notes) parts.push(`Invoice Notes: ${normalizedData.notes}`);
      break;

    case "doctor":
      parts.push(`Doctor Profile: ${normalizedData.name ?? "Unknown"}`);
      if (normalizedData.clinicName) parts.push(`Clinic/Office Name: ${normalizedData.clinicName}`);
      if (normalizedData.specialty) parts.push(`Dental Specialty: ${normalizedData.specialty}`);
      if (normalizedData.email) parts.push(`Contact Email: ${normalizedData.email}`);
      if (normalizedData.phone) parts.push(`Contact Phone: ${normalizedData.phone}`);
      if (normalizedData.preferences) parts.push(`Doctor Preferences: ${normalizedData.preferences}`);
      break;

    default:
      parts.push(`${entityType.toUpperCase()} Record: ${JSON.stringify(data)}`);
  }

  return parts.join(". ");
}

// ─── Embed + Upsert ───────────────────────────────────────────────────────────

export interface EmbedAndStoreResult {
  chunksProcessed: number;
  inputTokens: number;
}

/**
 * Takes an entity, generates embeddings for its text chunks,
 * and upserts them into the vector_embeddings table (with raw pgvector storage).
 */
export async function embedAndStore(
  labId: string,
  entityType: string,
  entityId: string,
  data: Record<string, unknown>
): Promise<EmbedAndStoreResult> {
  const rawText = buildEmbeddableText(entityType, data);
  const chunks = chunkText(rawText);

  if (chunks.length === 0) {
    return { chunksProcessed: 0, inputTokens: 0 };
  }

  const response = await embedTextsWithGemini(chunks, "RETRIEVAL_DOCUMENT");

  const inputTokens = response.inputTokens;

  // Upsert each chunk into DB — delete old chunks for this entity first,
  // then insert fresh ones (simpler than diff-patching)
  await prisma.vectorEmbedding.deleteMany({
    where: { labId, entityType, entityId },
  });

  // Store text content (the raw pgvector column is managed via raw SQL below)
  const records = chunks.map((content, idx) => ({
    labId,
    entityType,
    entityId,
    chunkIndex: idx,
    content,
  }));

  await prisma.vectorEmbedding.createMany({ data: records });

  // Write the actual vector data via raw SQL (pgvector extension)
  // We fetch the just-inserted IDs in order and update each.
  const inserted = await prisma.vectorEmbedding.findMany({
    where: { labId, entityType, entityId },
    orderBy: { chunkIndex: "asc" },
    select: { id: true, chunkIndex: true },
  });

  for (const row of inserted) {
    const embeddingData = response.embeddings[row.chunkIndex];
    if (!embeddingData) continue;
    const vectorLiteral = `[${embeddingData.join(",")}]`;
    await prisma.$executeRawUnsafe(
      `UPDATE vector_embeddings SET embedding = $1::vector WHERE id = $2`,
      vectorLiteral,
      row.id
    );
  }

  return { chunksProcessed: chunks.length, inputTokens };
}

// ─── Similarity Search ────────────────────────────────────────────────────────

export interface SearchResult {
  id: string;
  entityType: string;
  entityId: string;
  content: string;
  similarity: number;
}

/**
 * Perform a vector similarity search scoped to a specific lab.
 * Returns the top-k most relevant chunks.
 */
export async function searchSimilar(
  labId: string,
  query: string,
  topK = 5,
  entityTypeFilter?: string
): Promise<SearchResult[]> {
  // Embed the query
  const response = await embedTextsWithGemini([query], "RETRIEVAL_QUERY");

  const firstEmbedding = response.embeddings[0];
  if (!firstEmbedding) throw new Error("Embedding API returned no results for query");
  const queryVector = `[${firstEmbedding.join(",")}]`;
  const entityFilter = entityTypeFilter ? `AND entity_type = '${entityTypeFilter}'` : "";

  // Use pgvector cosine distance operator (<=>)
  type RawRow = { id: string; entity_type: string; entity_id: string; content: string; similarity: number };

  const results = (await prisma.$queryRawUnsafe<RawRow[]>(
    `
    SELECT id, entity_type, entity_id, content,
           1 - (embedding <=> $1::vector) AS similarity
    FROM vector_embeddings
    WHERE lab_id = $2 ${entityFilter}
    ORDER BY embedding <=> $1::vector
    LIMIT $3
    `,
    queryVector,
    labId,
    topK
  )) as RawRow[];

  return results.map((r: RawRow) => ({
    id: r.id,
    entityType: r.entity_type,
    entityId: r.entity_id,
    content: r.content,
    similarity: r.similarity,
  }));
}
