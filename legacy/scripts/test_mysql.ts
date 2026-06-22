import mysql from "mysql2/promise";

async function main() {
  const mysqlUrl = process.env.MONOLITH_DATABASE_URL;
  console.log("URL exists:", !!mysqlUrl);
  if (!mysqlUrl) return;

  try {
    const connection = await mysql.createConnection(mysqlUrl);
    console.log("✅ MySQL Connected successfully!");
    const [rows] = await connection.execute("SELECT 1 as result");
    console.log("Query result:", rows);
    await connection.end();
  } catch (err) {
    console.error("❌ MySQL Connection failed:", err);
  }
}

main().catch(console.error);
