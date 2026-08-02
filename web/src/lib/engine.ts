/**
 * The recommender, running in the browser.
 *
 * A TypeScript port of the user tower in src/model.py (a pre-norm causal
 * Transformer) plus the exact dot-product retrieval from src/api/recommender.py.
 * It reads the files written by scripts/export_web_model.py, so the site
 * needs no server. web/scripts/check-parity.mts verifies it against PyTorch.
 *
 * This file has no imports on purpose: Node can run it directly for the
 * parity check.
 */

export type Manifest = {
  dim: number;
  max_len: number;
  num_heads: number;
  num_layers: number;
  num_movies: number;
  tensors: Record<string, { offset: number; shape: number[] }>;
};

/** The catalog, column-wise, in the same order as the film vectors. */
export type Catalog = {
  genre_names: string[];
  movie_id: number[];
  title: (string | null)[];
  year: (number | null)[];
  genres: number[][];
  poster_path: (string | null)[];
  director: (string | null)[];
  tmdb_id: (number | null)[];
  cold_start: number[];
  search_rank: number[];
};

export type Scored = { index: number; score: number };

const LAYER_NORM_EPS = 1e-5;

/** IEEE 754 half floats to float32. */
export function decodeFloat16(bytes: ArrayBuffer): Float32Array {
  const halves = new Uint16Array(bytes);
  const out = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i++) {
    const h = halves[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    if (exponent === 0) {
      out[i] = sign * fraction * 2 ** -24;
    } else if (exponent === 31) {
      out[i] = fraction ? NaN : sign * Infinity;
    } else {
      out[i] = sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
    }
  }
  return out;
}

/** Abramowitz and Stegun 7.1.26, accurate to about 1.5e-7. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  return sign * (1 - poly * Math.exp(-a * a));
}

function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x * Math.SQRT1_2));
}

export class Engine {
  readonly manifest: Manifest;
  readonly catalog: Catalog;
  readonly size: number;
  private readonly weights: Float32Array;
  private readonly items: Int8Array;
  private readonly scales: Float32Array;
  private readonly idToIndex = new Map<number, number>();
  private readonly titlesLower: string[];

  constructor(
    manifest: Manifest,
    weights: Float32Array,
    items: Int8Array,
    scales: Float32Array,
    catalog: Catalog,
  ) {
    if (items.length !== manifest.num_movies * manifest.dim) {
      throw new Error("Film vectors do not match the manifest.");
    }
    this.manifest = manifest;
    this.weights = weights;
    this.items = items;
    this.scales = scales;
    this.catalog = catalog;
    this.size = manifest.num_movies;
    catalog.movie_id.forEach((id, index) => this.idToIndex.set(id, index));
    this.titlesLower = catalog.title.map((title) => (title ?? "").toLowerCase());
  }

  indexOf(movieId: number): number | undefined {
    return this.idToIndex.get(movieId);
  }

  /** Case-insensitive title search: prefix matches first, then by popularity. */
  search(query: string, limit = 8): number[] {
    const q = query.trim().toLowerCase();
    if (!q) {
      return [];
    }
    const rank = this.catalog.search_rank;
    const matches: number[] = [];
    for (let i = 0; i < this.titlesLower.length; i++) {
      if (this.titlesLower[i].includes(q)) {
        matches.push(i);
      }
    }
    const starts = (i: number) => (this.titlesLower[i].startsWith(q) ? 0 : 1);
    matches.sort((a, b) => starts(a) - starts(b) || rank[a] - rank[b]);
    return matches.slice(0, limit);
  }

  /**
   * Rank the whole catalog for a history of film indices, oldest first.
   * Films already in the history are left out.
   */
  recommend(history: number[], k: number): Scored[] {
    if (history.length === 0) {
      return [];
    }
    const { dim } = this.manifest;
    const user = this.encodeUser(history);
    const scores = new Float32Array(this.size);
    const items = this.items;
    for (let i = 0, base = 0; i < this.size; i++, base += dim) {
      let sum = 0;
      for (let j = 0; j < dim; j++) {
        sum += user[j] * items[base + j];
      }
      scores[i] = sum * this.scales[i];
    }
    for (const index of history) {
      scores[index] = -Infinity;
    }

    const top: Scored[] = [];
    for (let i = 0; i < this.size; i++) {
      const score = scores[i];
      if (top.length === k && score <= top[k - 1].score) {
        continue;
      }
      let at = top.length;
      while (at > 0 && top[at - 1].score < score) {
        at--;
      }
      top.splice(at, 0, { index: i, score });
      if (top.length > k) {
        top.pop();
      }
    }
    return top.filter((entry) => entry.score > -Infinity);
  }

  // ------------------------------------------------------------------
  // User tower
  // ------------------------------------------------------------------

  private tensor(name: string): Float32Array {
    const entry = this.manifest.tensors[name];
    const length = entry.shape.reduce((a, b) => a * b, 1);
    return this.weights.subarray(entry.offset, entry.offset + length);
  }

  /** out[r] = x[r] @ W^T + b for rows from `start`; W is [outDim, inDim]. */
  private linear(
    x: Float32Array,
    rows: number,
    inDim: number,
    w: Float32Array,
    b: Float32Array,
    outDim: number,
    start = 0,
  ): Float32Array {
    const out = new Float32Array(rows * outDim);
    for (let r = start; r < rows; r++) {
      const xBase = r * inDim;
      const outBase = r * outDim;
      for (let o = 0, wBase = 0; o < outDim; o++, wBase += inDim) {
        let sum = b[o];
        for (let i = 0; i < inDim; i++) {
          sum += x[xBase + i] * w[wBase + i];
        }
        out[outBase + o] = sum;
      }
    }
    return out;
  }

  private layerNorm(x: Float32Array, rows: number, w: Float32Array, b: Float32Array) {
    const { dim } = this.manifest;
    const out = new Float32Array(x.length);
    for (let r = 0; r < rows; r++) {
      const base = r * dim;
      let mean = 0;
      for (let j = 0; j < dim; j++) {
        mean += x[base + j];
      }
      mean /= dim;
      let variance = 0;
      for (let j = 0; j < dim; j++) {
        const d = x[base + j] - mean;
        variance += d * d;
      }
      const inv = 1 / Math.sqrt(variance / dim + LAYER_NORM_EPS);
      for (let j = 0; j < dim; j++) {
        out[base + j] = (x[base + j] - mean) * inv * w[j] + b[j];
      }
    }
    return out;
  }

  /** The user vector: the Transformer's state at the most recent film. */
  private encodeUser(history: number[]): Float32Array {
    const { dim, max_len: maxLen, num_heads: heads, num_layers: layers } = this.manifest;
    const recent = history.slice(-maxLen);
    const length = recent.length;
    const headDim = dim / heads;
    const attnScale = 1 / Math.sqrt(headDim);

    // Film vectors scaled by sqrt(dim), plus right-aligned position embeddings.
    const position = this.tensor("position");
    const x = new Float32Array(length * dim);
    const inputScale = Math.sqrt(dim);
    for (let t = 0; t < length; t++) {
      const itemBase = recent[t] * dim;
      const scale = this.scales[recent[t]] * inputScale;
      const posBase = (maxLen - length + t) * dim;
      for (let j = 0; j < dim; j++) {
        x[t * dim + j] = this.items[itemBase + j] * scale + position[posBase + j];
      }
    }

    for (let layer = 0; layer < layers; layer++) {
      const name = (part: string) => this.tensor(`l${layer}.${part}`);
      // Only the last position is read after the final layer, so that layer
      // computes just that row (it still attends over every earlier film).
      const start = layer === layers - 1 ? length - 1 : 0;

      // Self-attention block: x = x + attention(norm1(x)), causal.
      const normed = this.layerNorm(x, length, name("norm1.w"), name("norm1.b"));
      const qkv = this.linear(normed, length, dim, name("attn.in.w"), name("attn.in.b"), 3 * dim);
      const attended = new Float32Array(length * dim);
      const weights = new Float32Array(length);
      for (let h = 0; h < heads; h++) {
        const qOffset = h * headDim;
        const kOffset = dim + h * headDim;
        const vOffset = 2 * dim + h * headDim;
        for (let t = start; t < length; t++) {
          let max = -Infinity;
          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let j = 0; j < headDim; j++) {
              dot += qkv[t * 3 * dim + qOffset + j] * qkv[s * 3 * dim + kOffset + j];
            }
            weights[s] = dot * attnScale;
            max = Math.max(max, weights[s]);
          }
          let total = 0;
          for (let s = 0; s <= t; s++) {
            weights[s] = Math.exp(weights[s] - max);
            total += weights[s];
          }
          for (let s = 0; s <= t; s++) {
            const a = weights[s] / total;
            for (let j = 0; j < headDim; j++) {
              attended[t * dim + qOffset + j] += a * qkv[s * 3 * dim + vOffset + j];
            }
          }
        }
      }
      const projected = this.linear(
        attended, length, dim, name("attn.out.w"), name("attn.out.b"), dim, start,
      );
      for (let i = start * dim; i < length * dim; i++) {
        x[i] += projected[i];
      }

      // Feed-forward block: x = x + ff2(gelu(ff1(norm2(x)))).
      const normed2 = this.layerNorm(x, length, name("norm2.w"), name("norm2.b"));
      const hidden = this.linear(
        normed2, length, dim, name("ff1.w"), name("ff1.b"), 4 * dim, start,
      );
      for (let i = start * 4 * dim; i < hidden.length; i++) {
        hidden[i] = gelu(hidden[i]);
      }
      const ff = this.linear(hidden, length, 4 * dim, name("ff2.w"), name("ff2.b"), dim, start);
      for (let i = start * dim; i < length * dim; i++) {
        x[i] += ff[i];
      }
    }

    const last = x.slice((length - 1) * dim);
    return this.layerNorm(last, 1, this.tensor("out.w"), this.tensor("out.b"));
  }
}
