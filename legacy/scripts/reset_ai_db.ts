import { Pool } from "pg";
import { execSync } from "child_process";

async function main() {
  const connectionString = process.env.DATABASE_URL;

  if (!connectionString) {
    throw new Error("DATABASE_URL is not set");
  }

  console.log("Resetting AI PostgreSQL schema...");
  const pool = new Pool({ connectionString });

  try {
    await pool.query("DROP SCHEMA IF EXISTS public CASCADE;");
    await pool.query("CREATE SCHEMA public;");
    await pool.query("CREATE EXTENSION IF NOT EXISTS vector;");
    console.log("Schema reset completed.");
  } finally {
    await pool.end();
  }

  console.log("Applying Prisma schema...");
  execSync("bunx prisma db push", { stdio: "inherit" });
  console.log("AI DB reset is ready.");
}

main().catch((error) => {
  console.error("Failed to reset AI DB:", error);
  process.exit(1);
});
