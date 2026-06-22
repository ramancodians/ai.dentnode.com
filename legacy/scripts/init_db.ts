import { Pool } from "pg";
import { execSync } from "child_process";

async function main() {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error("DATABASE_URL is not set");
  }

  console.log("Connecting to PostgreSQL...");
  const pool = new Pool({ connectionString });

  try {
    console.log("Enabling pgvector extension...");
    await pool.query("CREATE EXTENSION IF NOT EXISTS vector;");
    console.log("Extension enabled.");
  } catch (err) {
    console.error("Error creating extension:", err);
  } finally {
    await pool.end();
  }

  console.log("Pushing Prisma schema...");
  execSync("bunx prisma db push", { stdio: "inherit" });
  console.log("Schema pushed successfully.");
}

main().catch(console.error);
