/**
 * Checks the TypeScript engine against PyTorch.
 *
 *   python -m scripts.export_web_model      (from the repository root)
 *   node scripts/check-parity.mts            (from web/)
 *
 * The export writes the exact top results PyTorch gives on full-precision
 * vectors. The browser uses int8 film vectors and float16 weights, so the
 * lists are compared by overlap rather than for exact equality.
 */

import { readFileSync } from "node:fs";
import { Engine, decodeFloat16 } from "../src/lib/engine.ts";

const model = new URL("../public/model/", import.meta.url);
const buffer = (name: string) => {
  const file = readFileSync(new URL(name, model));
  return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
};
const json = (url: URL) => JSON.parse(readFileSync(url, "utf8"));

const engine = new Engine(
  json(new URL("manifest.json", model)),
  decodeFloat16(buffer("weights.bin")),
  new Int8Array(buffer("items.bin")),
  new Float32Array(buffer("scales.bin")),
  json(new URL("movies.json", model)),
);

type Case = { movie_ids: number[]; top: number[]; scores: number[] };
const cases: Case[] = json(new URL("../../artifacts/web_parity.json", import.meta.url));

const K = 12;
let failed = false;
for (const testCase of cases) {
  const history = testCase.movie_ids.map((id) => engine.indexOf(id)!);
  const started = performance.now();
  const got = engine.recommend(history, K);
  const ms = performance.now() - started;

  const ids = got.map((entry) => engine.catalog.movie_id[entry.index]);
  const expected = new Set(testCase.top.slice(0, K));
  const overlap = ids.filter((id) => expected.has(id)).length;
  const scoreError = Math.abs(got[0].score - testCase.scores[testCase.top.indexOf(ids[0])]);
  const ok = overlap >= K - 1 && ids[0] === testCase.top[0];
  failed ||= !ok;
  console.log(
    `${ok ? "ok  " : "FAIL"} history ${String(history.length).padStart(2)}  ` +
      `top-${K} overlap ${overlap}/${K}  top-1 ${ids[0] === testCase.top[0] ? "same" : "differs"}  ` +
      `score error ${scoreError.toFixed(4)}  ${ms.toFixed(0)} ms`,
  );
}
process.exit(failed ? 1 : 0);
