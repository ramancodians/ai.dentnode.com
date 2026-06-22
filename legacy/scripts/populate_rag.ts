import mysql from "mysql2/promise";
import { embedAndStore } from "../src/services/embedder";
import prisma from "../src/db/client";

const BATCH_SIZE = 250;

async function processBatches<T extends Record<string, unknown>>(
  connection: mysql.Connection,
  query: string,
  label: string,
  processRow: (row: T) => Promise<void>
) {
  let offset = 0;

  while (true) {
    const batchSize = Math.max(1, Number(BATCH_SIZE));
    const safeOffset = Math.max(0, Number(offset));
    const pagedQuery = `${query}\nLIMIT ${batchSize} OFFSET ${safeOffset}`;
    const [rows] = await connection.query(pagedQuery);
    const batch = rows as T[];

    if (batch.length === 0) {
      break;
    }

    for (const row of batch) {
      await processRow(row);
    }

    offset += batch.length;
    console.log(`   Processed ${offset} ${label}...`);

    if (batch.length < BATCH_SIZE) {
      break;
    }
  }
}

/**
 * Migration script to populate the RAG vector database from the monolith MySQL database.
 * Fetches Doctors, Entries (Cases), and Invoices and generates embeddings for them.
 */
async function main() {
  const mysqlUrl = process.env.MONOLITH_DATABASE_URL;
  if (!mysqlUrl) {
    console.error("❌ MONOLITH_DATABASE_URL not found in environment.");
    process.exit(1);
  }

  console.log("🔗 Connecting to Monolith MySQL Database...");
  const connection = await mysql.createConnection(mysqlUrl);

  try {
    // 1. Migrate Doctors
    console.log("\n👨‍⚕️ Migrating Doctors...");
    await processBatches<any>(
      connection,
      `SELECT id, first_name, last_name, clinic_name, full_address, email, phone, lab_id
       FROM Doctors
       WHERE partership_status != 'DELETED' OR partership_status IS NULL
        ORDER BY id`,
      "doctors",
      async (doc) => {
        console.log(`   Embedding Doctor: ${doc.first_name} ${doc.last_name || ""} (${doc.id})`);
        await embedAndStore(doc.lab_id, "doctor", doc.id, {
          name: `${doc.first_name} ${doc.last_name || ""}`.trim(),
          clinicName: doc.clinic_name,
          email: doc.email,
          phone: doc.phone,
          address: doc.full_address,
        });
      }
    );

    // 2. Migrate Entries (Cases)
    console.log("\n📦 Migrating Cases (Entries)...");
    await processBatches<any>(
      connection,
      `SELECT
          e.id,
          e.status,
          e.notes,
          e.created_at,
          e.lab_id,
          e.doctor_id,
          e.case_custom_id,
          e.epxected_delivery_date,
          e.expected_delivery_slot,
          e.delivery_type,
          e.delivery_notes,
          d.first_name as doctor_first,
          d.last_name as doctor_last,
          p.first_name as patient_first,
          p.last_name as patient_last,
          GROUP_CONCAT(DISTINCT pr.name ORDER BY pr.name SEPARATOR ', ') as product_names,
          GROUP_CONCAT(
            DISTINCT CASE
              WHEN w.shade IS NOT NULL AND w.shade != '' AND w.shade != 'NA' THEN w.shade
              ELSE NULL
            END
            ORDER BY w.shade SEPARATOR ', '
          ) as shades,
          GROUP_CONCAT(DISTINCT dept.name ORDER BY dept.name SEPARATOR ', ') as department_names
       FROM Entry e
       LEFT JOIN Doctors d ON e.doctor_id = d.id
       LEFT JOIN Patients p ON p.entry_id = e.id
       LEFT JOIN Work w ON w.entryId = e.id
       LEFT JOIN Products pr ON w.productId = pr.id
       LEFT JOIN Departments dept ON w.department_id = dept.id
       WHERE e.deleted = 0
       GROUP BY
         e.id,
         e.status,
         e.notes,
         e.created_at,
         e.lab_id,
         e.doctor_id,
         e.case_custom_id,
         e.epxected_delivery_date,
         e.expected_delivery_slot,
         e.delivery_type,
         e.delivery_notes,
         d.first_name,
         d.last_name,
         p.first_name,
         p.last_name
       ORDER BY e.created_at DESC`,
      "cases",
      async (entry) => {
        console.log(`   Embedding Case: ${entry.case_custom_id || entry.id}`);
        await embedAndStore(entry.lab_id, "case", entry.id, {
          id: entry.case_custom_id || entry.id,
          case_custom_id: entry.case_custom_id,
          status: entry.status,
          doctorName: `${entry.doctor_first} ${entry.doctor_last || ""}`.trim(),
          patientName: `${entry.patient_first || ""} ${entry.patient_last || ""}`.trim(),
          dueDate: entry.epxected_delivery_date,
          expectedDeliverySlot: entry.expected_delivery_slot,
          deliveryType: entry.delivery_type,
          deliveryNotes: entry.delivery_notes,
          productNames: entry.product_names,
          shades: entry.shades,
          departmentNames: entry.department_names,
          notes: entry.notes,
          createdAt: entry.created_at,
        });
      }
    );

    // 3. Migrate Invoices
    console.log("\n🧾 Migrating Invoices...");
    await processBatches<any>(
      connection,
      `SELECT i.id, i.total_amount, i.bill_amount, i.due_date, i.doctor_id, d.lab_id,
              d.first_name as doctor_first, d.last_name as doctor_last
       FROM Invoices i
       JOIN Doctors d ON i.doctor_id = d.id
       WHERE i.is_active = 1
        ORDER BY i.due_date DESC, i.id DESC`,
      "invoices",
      async (inv) => {
        console.log(`   Embedding Invoice: #${inv.id}`);
        await embedAndStore(inv.lab_id, "invoice", String(inv.id), {
          id: String(inv.id),
          amount: inv.total_amount || inv.bill_amount,
          dueDate: inv.due_date,
          doctorName: `${inv.doctor_first} ${inv.doctor_last || ""}`.trim(),
          status: inv.total_amount > 0 ? "Active" : "Draft",
        });
      }
    );

    console.log("\n✅ Migration completed successfully.");
  } catch (error) {
    console.error("❌ Error during migration:", error);
  } finally {
    await connection.end();
    await prisma.$disconnect();
  }
}

main().catch(console.error);
