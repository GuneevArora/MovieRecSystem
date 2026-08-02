/**
 * Loads the in-browser engine once, on first use, and reports download
 * progress. The files come from /model/, written by
 * scripts/export_web_model.py.
 */

import { type Catalog, Engine, type Manifest, decodeFloat16 } from "@/lib/engine";

type Listener = (fraction: number) => void;

const listeners = new Set<Listener>();
let fraction = 0;
let loading: Promise<Engine> | null = null;

/** Subscribe to download progress, 0 to 1. Returns an unsubscribe function. */
export function onModelProgress(listener: Listener): () => void {
  listeners.add(listener);
  listener(fraction);
  return () => {
    listeners.delete(listener);
  };
}

function report(value: number) {
  fraction = value;
  listeners.forEach((listener) => listener(value));
}

async function download(name: string, onBytes: (count: number) => void) {
  const response = await fetch(`/model/${name}`);
  if (!response.ok || !response.body) {
    throw new Error(`Could not load ${name} (${response.status})`);
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    size += value.length;
    onBytes(value.length);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function load(): Promise<Engine> {
  const manifest: Manifest & { bytes: Record<string, number> } = await (
    await fetch("/model/manifest.json")
  ).json();
  const total = Object.values(manifest.bytes).reduce((a, b) => a + b, 0);
  let loaded = 0;
  const count = (bytes: number) => {
    loaded += bytes;
    // Hold at 99% until the engine is built.
    report(Math.min(0.99, loaded / total));
  };

  const [weights, items, scales, movies] = await Promise.all(
    ["weights.bin", "items.bin", "scales.bin", "movies.json"].map((name) =>
      download(name, count),
    ),
  );
  const catalog: Catalog = JSON.parse(new TextDecoder().decode(movies));
  const engine = new Engine(
    manifest,
    decodeFloat16(weights.buffer as ArrayBuffer),
    new Int8Array(items.buffer as ArrayBuffer),
    new Float32Array(scales.buffer as ArrayBuffer),
    catalog,
  );
  report(1);
  return engine;
}

/** The engine, downloading it first if needed. Safe to call repeatedly. */
export function getEngine(): Promise<Engine> {
  if (!loading) {
    loading = load().catch((error) => {
      // Let a later call retry after a failed download.
      loading = null;
      report(0);
      throw error;
    });
  }
  return loading;
}
