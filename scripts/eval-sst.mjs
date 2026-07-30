/**
 * Large-scale accuracy test against Stanford Sentiment Treebank ground truth.
 *
 *   node scripts/prepare-sst.mjs                 # once
 *   node scripts/eval-sst.mjs --n 500 --model qwen3.5:9b-gpu
 *
 * Flags: --n --refine --concurrency --model --seed --prompt --distinct --out
 */
import { readFileSync, appendFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { rankByPrompt, createComparator } from "../dist/index.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith("--") ? args[i + 1] : fallback;
};
const has = (name) => args.includes(`--${name}`);

const N = Number(flag("n", 100));
const SEED = Number(flag("seed", 1));
const MODEL = flag("model", "qwen3.5:9b-gpu");
const CONCURRENCY = Number(flag("concurrency", 8));
const REFINE = flag("refine", undefined);
const PROMPT = flag("prompt", "the most positive and enthusiastic sentiment about the film");
const OUT = flag("out", null);

const mulberry = (a) => () => (a = a + 0x6d2b79f5 | 0, ((t) => (t = Math.imul(t ^ t >>> 15, t | 1), t ^= t + Math.imul(t ^ t >>> 7, t | 61), ((t ^ t >>> 14) >>> 0) / 4294967296))(a));

const all = readFileSync(join(ROOT, "data", "sst-sentences.jsonl"), "utf8")
  .split("\n").filter(Boolean).map((l) => JSON.parse(l))
  // Very short fragments ("Abysmally pathetic") are unrepresentative and ambiguous out of context.
  .filter((r) => r.text.split(/\s+/).length >= 6);

const rand = mulberry(SEED);
const pool = all.slice();
for (let i = pool.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [pool[i], pool[j]] = [pool[j], pool[i]];
}

let sample;
if (has("distinct")) {
  // One item per distinct score => no ground-truth ties, so tau is unambiguous.
  const seen = new Set();
  sample = [];
  for (const r of pool) {
    if (seen.has(r.score)) continue;
    seen.add(r.score);
    sample.push(r);
    if (sample.length === N) break;
  }
} else {
  sample = pool.slice(0, N);
}

if (sample.length < N) {
  // Only ~82 distinct score values exist, so --distinct cannot reach large n.
  console.error(`warning: only ${sample.length} items available (asked for ${N})` +
    `${has("distinct") ? "; --distinct is limited by the number of distinct scores" : ""}`);
}

const items = sample.map((r) => r.text);
const truth = sample.map((r) => r.score);

/** Kendall tau-b (handles ties in the ground truth) and Spearman rho. */
function correlations(predRankOf, truthScores) {
  const n = truthScores.length;
  let C = 0, D = 0, Tx = 0, Ty = 0;
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++) {
      const dt = truthScores[a] - truthScores[b];
      const dp = predRankOf[b] - predRankOf[a]; // lower rank index = better = higher score
      if (dt === 0 && dp === 0) continue;
      if (dt === 0) { Ty++; continue; }
      if (dp === 0) { Tx++; continue; }
      if (Math.sign(dt) === Math.sign(dp)) C++; else D++;
    }
  const tau = (C - D) / Math.sqrt((C + D + Tx) * (C + D + Ty));

  const rankOfTruth = truthScores.map((_, i) => i)
    .sort((a, b) => truthScores[b] - truthScores[a]);
  const truthRank = new Array(n);
  rankOfTruth.forEach((id, k) => (truthRank[id] = k));
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (predRankOf[i] - truthRank[i]) ** 2;
  const rho = 1 - (6 * d2) / (n * (n * n - 1));
  return { tau, rho, concordant: C, discordant: D };
}

/** Pairwise accuracy bucketed by true score gap: separates algorithm error from genuine coin flips. */
function accuracyByGap(predRankOf, truthScores) {
  const buckets = [[0, 0.05], [0.05, 0.15], [0.15, 0.3], [0.3, 0.6], [0.6, 1.01]];
  const stats = buckets.map(() => ({ right: 0, total: 0 }));
  const n = truthScores.length;
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++) {
      const gap = Math.abs(truthScores[a] - truthScores[b]);
      if (gap === 0) continue;
      const k = buckets.findIndex(([lo, hi]) => gap >= lo && gap < hi);
      if (k < 0) continue;
      const better = truthScores[a] > truthScores[b] ? a : b;
      const worse = better === a ? b : a;
      stats[k].total++;
      if (predRankOf[better] < predRankOf[worse]) stats[k].right++;
    }
  return buckets.map(([lo, hi], k) => ({ range: `${lo.toFixed(2)}-${hi === 1.01 ? "1.00" : hi.toFixed(2)}`, ...stats[k] }));
}

const sortedTruth = truth.slice().sort((a, b) => b - a);

const topK = (order, k) => {
  const trueTop = new Set(truth.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i));
  return order.slice(0, k).filter((i) => trueTop.has(i)).length;
};

/**
 * Tie-tolerant top-k: credits any predicted item whose true score matches or
 * beats the k-th highest true score. With ~82 score levels, the "true top k"
 * set is arbitrary among tied items, so plain top-k understates performance.
 */
const topKTieTolerant = (order, k) =>
  order.slice(0, k).filter((i) => truth[i] >= sortedTruth[k - 1]).length;

/** Mean true score of the predicted top-k vs the best achievable and the baseline. */
const topKQuality = (order, k) => {
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return {
    got: mean(order.slice(0, k).map((i) => truth[i])),
    best: mean(sortedTruth.slice(0, k)),
    baseline: mean(truth),
  };
};

console.log(`SST eval | n=${N} model=${MODEL} concurrency=${CONCURRENCY} refine=${REFINE ?? "default"} seed=${SEED}${has("distinct") ? " distinct-scores" : ""}`);
console.log(`prompt: "${PROMPT}"`);
const expected = Math.round(N * Math.log2(N) - N + 1);
console.log(`expecting ~${expected} merge-sort comparisons\n`);

const comparator = createComparator({ provider: "ollama", model: MODEL });
let last = 0;
const started = Date.now();
const result = await rankByPrompt(items, {
  prompt: PROMPT,
  comparator,
  concurrency: CONCURRENCY,
  refinementBudget: REFINE === undefined ? undefined : Number(REFINE),
  onProgress: ({ comparisons, phase }) => {
    if (comparisons - last >= 25) {
      last = comparisons;
      const el = (Date.now() - started) / 1000;
      process.stderr.write(`\r${phase}: ${comparisons} comparisons, ${el.toFixed(0)}s, ${(comparisons / el).toFixed(1)}/s   `);
    }
  },
});
const elapsed = (Date.now() - started) / 1000;
process.stderr.write("\r".padEnd(70) + "\r");

const predRankOf = new Array(N);
result.order.forEach((id, k) => (predRankOf[id] = k));

const { tau, rho, concordant, discordant } = correlations(predRankOf, truth);
console.log(`comparisons  ${result.comparisons}  (${(result.comparisons / (N * (N - 1) / 2) * 100).toFixed(1)}% of all ${N * (N - 1) / 2} pairs)`);
console.log(`wall time    ${elapsed.toFixed(1)}s  (${(result.comparisons / elapsed).toFixed(1)} comparisons/s, ${(elapsed / result.comparisons * 1000).toFixed(0)}ms effective/call)`);
console.log(`Kendall tau  ${tau.toFixed(4)}   (concordant ${concordant}, discordant ${discordant})`);
console.log(`Spearman rho ${rho.toFixed(4)}`);
console.log(`\ntop-k: strict = exact set overlap; tie-ok = true score >= k-th best (ties are arbitrary)`);
console.log(`         mean true score of picks vs best achievable vs list average`);
for (const k of [10, 25, 50].filter((k) => k <= N / 2)) {
  const q = topKQuality(result.order, k);
  console.log(`top-${String(k).padEnd(3)}  strict ${String(topK(result.order, k)).padStart(2)}/${k}   ` +
    `tie-ok ${String(topKTieTolerant(result.order, k)).padStart(2)}/${k}   ` +
    `score ${q.got.toFixed(3)} (best ${q.best.toFixed(3)}, avg ${q.baseline.toFixed(3)})`);
}

console.log(`\npairwise accuracy by true score gap:`);
for (const b of accuracyByGap(predRankOf, truth))
  if (b.total) console.log(`  gap ${b.range}  ${(b.right / b.total * 100).toFixed(1)}%  (${b.right}/${b.total})`);

const DUMP = flag("dump", null);
if (DUMP) {
  // Persist the ranking so post-hoc analysis never requires re-running the sort.
  writeFileSync(DUMP, JSON.stringify({ n: N, model: MODEL, seed: SEED, prompt: PROMPT,
    comparisons: result.comparisons, elapsed,
    ranked: result.order.map((id, rank) => ({ rank: rank + 1, truth: truth[id], score: result.scores[id], text: items[id] })),
  }, null, 1));
  console.log(`\ndumped ranking to ${DUMP}`);
}

if (OUT) {
  appendFileSync(OUT, JSON.stringify({ n: N, model: MODEL, seed: SEED, refine: REFINE ?? null,
    concurrency: CONCURRENCY, comparisons: result.comparisons, elapsed, tau, rho }) + "\n");
  console.log(`\nappended to ${OUT}`);
}
