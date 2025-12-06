// import { PrismaClient } from "@prisma/client";
import { PrismaClient } from "./generated/prisma/client";

// Prevent creating multiple PrismaClient instances in dev/hot-reload
const globalForPrisma = global as unknown as { prisma?: PrismaClient };

export const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({
    log: ["error", "warn"],
  });

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
