FROM oven/bun:1 AS base
WORKDIR /app

# Install dependencies (only required ones)
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Generate Prisma Client & prepare source
COPY prisma ./prisma
COPY prisma.config.ts ./
RUN bunx prisma generate

# Copy source code
COPY src ./src
COPY tsconfig.json ./

# Final runtime stage
FROM oven/bun:1-slim
WORKDIR /app

# Copy the built dependencies and prisma client from the base image
COPY --from=base /app/node_modules ./node_modules
COPY --from=base /app/package.json ./
COPY --from=base /app/prisma ./prisma

# Copy source code
COPY src ./src

# Set environment
ENV NODE_ENV=production
ENV PORT=8000

USER bun
EXPOSE 8000
CMD ["bun", "run", "src/index.ts"]
