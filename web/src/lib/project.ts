/**
 * Offline results, measured on the held-out test split with
 * `python -m src.evaluate --split test` (see results/metrics.json).
 * Leave-one-out, next-item, ranked against the full catalog.
 */
export const metrics = [
  { value: "0.237", label: "Recall@10" },
  { value: "0.135", label: "NDCG@10" },
  { value: "5.8×", label: "vs popularity" },
  { value: "15.5M", label: "Interactions" },
];

export const catalogSize = 54732;

export const stack = [
  "PyTorch",
  "FastAPI",
  "Next.js",
  "Docker",
  "GitHub Actions",
  "Terraform",
];

export const steps = [
  {
    number: "01",
    title: "Read the sequence",
    body: "A causal Transformer reads the films you picked, in order, and compresses them into a single taste vector. No account or user id is needed.",
  },
  {
    number: "02",
    title: "Score every film",
    body: "That vector is compared against all 54,732 films in the catalog with a dot product. Nothing is pre-filtered, so niche picks can surface.",
  },
  {
    number: "03",
    title: "Serve in milliseconds",
    body: "A FastAPI service batches concurrent requests into one forward pass and returns the top matches, minus what you have already seen.",
  },
];

export const presets = [
  { label: "Mind-bending sci-fi", ids: [2571, 79132, 541, 1196] },
  { label: "Animated favourites", ids: [1, 3114, 4886, 6377] },
  { label: "Classic horror", ids: [1258, 1219, 593, 1387] },
];

export const repoUrl = "https://github.com/GuneevArora/MovieRecSystem";
