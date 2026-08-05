/**
 * Builds the static site for hosting under a sub-path and lays it out so the
 * files sit at that path: dist/<base path>/...
 */

import { execSync } from "node:child_process";
import { cpSync, rmSync } from "node:fs";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "/movierecdemo";

execSync("npx next build", {
  stdio: "inherit",
  env: { ...process.env, NEXT_PUBLIC_BASE_PATH: basePath },
});
rmSync("dist", { recursive: true, force: true });
cpSync("out", `dist${basePath}`, { recursive: true });
console.log(`Built dist${basePath}`);
