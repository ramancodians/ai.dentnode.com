import prisma from "../src/db/client";

async function main() {
  const counts = await prisma.vectorEmbedding.groupBy({
    by: ['entityType'],
    _count: {
       id: true
    }
  });

  console.log("Vector Population Status:");
  console.table(counts.map(c => ({ Entity: c.entityType, Count: c._count.id })));
}

main().catch(console.error).finally(() => prisma.$disconnect());
