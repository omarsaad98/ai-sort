/**
 * Post-hoc analysis of a ranking dumped by eval-sst.mjs --dump.
 *
 * Reports both selection measures side by side, since they diverge sharply
 * when the top of the list is tightly clustered:
 *   precision - exact overlap with an optimal k-subset
 *   regret    - (best - got) / (best - random) on mean true label; 0 optimal, 1 random
 *
 * Run: node scripts/analyze-dump.mjs data/rank-500.json
 */
import { readFileSync } from "node:fs";

const path = process.argv[2] ?? "data/rank-500.json";
const dump = JSON.parse(readFileSync(path, "utf8"));
const truth = dump.ranked.map((r) => r.truth); // index = predicted rank - 1
const sortedDesc = truth.slice().sort((a, b) => b - a);
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const listMean = mean(truth);

console.log(`${path}: n=${dump.n}, ${dump.comparisons} comparisons, model ${dump.model}`);
console.log(`list mean label ${listMean.toFixed(4)}\n`);
console.log("  k   precision   tie-ok   mean label   best      regret");
for (const k of [5, 10, 25, 50, 100].filter((k) => k <= dump.n / 2)) {
  const picked = truth.slice(0, k);
  const optimal = sortedDesc.slice(0, k);
  // Optimal k-subset is not unique when labels tie; credit any pick that
  // matches or beats the k-th best label.
  const threshold = sortedDesc[k - 1];
  const tieOk = picked.filter((t) => t >= threshold).length;
  // Strict precision needs multiset overlap against the optimal labels.
  const pool = optimal.slice();
  let strict = 0;
  for (const t of picked) {
    const at = pool.indexOf(t);
    if (at >= 0) { strict++; pool.splice(at, 1); }
  }
  const got = mean(picked), best = mean(optimal);
  const regret = (best - got) / (best - listMean);
  console.log(`${String(k).padStart(4)}   ${String(strict).padStart(3)}/${String(k).padEnd(4)} ` +
    `${String(tieOk).padStart(4)}/${String(k).padEnd(4)} ` +
    `${got.toFixed(4).padStart(9)}   ${best.toFixed(4)}   ${regret.toFixed(4).padStart(7)}`);
}
console.log(`\nregret 0 = optimal subset, 1 = no better than a random subset`);
