/**
 * Does the user actually need the top-k *ordered*, or just a good *subset*?
 *
 * These are different objectives with very different costs. This simulation
 * measures both, with ground truth known and no LLM calls:
 *
 *   strict precision  - |selected ∩ true top-k| / k. Punishes swapping two
 *                       near-identical items, which the user may not care about.
 *   normalized regret - (best - got) / (best - random), where each is the mean
 *                       true strength of a k-subset. 0 = optimal selection,
 *                       1 = no better than picking at random. Near-tie
 *                       confusions cost almost nothing here, which is the point.
 *
 * Run: node scripts/sim-topk.mjs [--n 200] [--k 10] [--trials 20] [--spread 4]
 */
const args = process.argv.slice(2);
const num = (name, d) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : d;
};
const N = num("n", 200), K = num("k", 10), TRIALS = num("trials", 20), SPREAD = num("spread", 4);

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const mulberry = (a) => () => (a = a + 0x6d2b79f5 | 0, ((t) => (t = Math.imul(t ^ t >>> 15, t | 1), t ^= t + Math.imul(t ^ t >>> 7, t | 61), ((t ^ t >>> 14) >>> 0) / 4294967296))(a));

// Item 0 is strongest. Adjacent items are nearly indistinguishable to the judge.
const truth = Array.from({ length: N }, (_, i) => SPREAD * (N - 1 - i) / (N - 1));

const PRIOR = 0.5;
function fitBT(edges, n, iters = 120) {
  const wins = new Map(), opps = Array.from({ length: n }, () => new Set());
  const key = (a, b) => a * n + b;
  for (const [w, l] of edges) {
    wins.set(key(w, l), (wins.get(key(w, l)) ?? 0) + 1);
    opps[w].add(l); opps[l].add(w);
  }
  let g = new Array(n).fill(1);
  for (let t = 0; t < iters; t++) {
    const next = new Array(n);
    for (let i = 0; i < n; i++) {
      if (opps[i].size === 0) { next[i] = g[i]; continue; }
      let numer = PRIOR, denom = (2 * PRIOR) / (g[i] + 1);
      for (const j of opps[i]) {
        const wij = wins.get(key(i, j)) ?? 0, wji = wins.get(key(j, i)) ?? 0;
        if (wij + wji === 0) continue;
        numer += wij; denom += (wij + wji) / (g[i] + g[j]);
      }
      next[i] = denom > 0 ? numer / denom : g[i];
      if (!Number.isFinite(next[i]) || next[i] <= 0) next[i] = g[i];
    }
    const lg = next.reduce((s, v) => s + Math.log(v > 0 ? v : 1e-9), 0);
    const gm = Math.exp(lg / n);
    g = next.map((v) => v / gm);
  }
  return g;
}
const rankOf = (scores) => Array.from({ length: scores.length }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);

function makeJudge(rand) {
  const fixed = new Map(); // one fixed verdict per pair, matching the real engine's memoization
  let calls = 0;
  const judge = (i, j) => {
    const k = i < j ? `${i}:${j}` : `${j}:${i}`;
    if (!fixed.has(k)) {
      const [lo, hi] = i < j ? [i, j] : [j, i];
      fixed.set(k, rand() < sigmoid(truth[lo] - truth[hi]) ? lo : hi);
      calls++;
    }
    return fixed.get(k);
  };
  return { judge, count: () => calls, seen: (i, j) => fixed.has(i < j ? `${i}:${j}` : `${j}:${i}`) };
}
const push = (edges, j, a, b) => { const w = j(a, b); edges.push([w, w === a ? b : a]); };

// ---- strategies -----------------------------------------------------------

/** Current pipeline: merge sort schedules queries, then fit globally. */
function fullSort(J, edges) {
  const ms = (ids) => {
    if (ids.length <= 1) return ids;
    const mid = ids.length >> 1;
    const L = ms(ids.slice(0, mid)), R = ms(ids.slice(mid)), out = [];
    let i = 0, jx = 0;
    while (i < L.length && jx < R.length) {
      const w = J.judge(L[i], R[jx]);
      edges.push([w, w === L[i] ? R[jx] : L[i]]);
      out.push(w === L[i] ? L[i++] : R[jx++]);
    }
    while (i < L.length) out.push(L[i++]);
    while (jx < R.length) out.push(R[jx++]);
    return out;
  };
  ms(Array.from({ length: N }, (_, i) => i));
}

/** Swiss-style: r rounds of pairings, each item plays ~r opponents. Fully parallel per round. */
function swiss(J, edges, rounds, rand) {
  for (let r = 0; r < rounds; r++) {
    // Round 1 is random; later rounds pair items of similar current strength,
    // which spends comparisons where the ordering is actually contested.
    let order;
    if (r === 0) {
      order = Array.from({ length: N }, (_, i) => i);
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    } else order = rankOf(fitBT(edges, N));
    for (let i = 0; i + 1 < order.length; i += 2) {
      if (!J.seen(order[i], order[i + 1])) push(edges, J.judge, order[i], order[i + 1]);
    }
  }
}

/** Selection: cheap screening, then settle the contested boundary exhaustively. */
function selectTopK(J, edges, rounds, candidateMult, rand) {
  swiss(J, edges, rounds, rand);
  const order = rankOf(fitBT(edges, N));
  const cands = order.slice(0, Math.min(N, K * candidateMult));
  for (let a = 0; a < cands.length; a++)
    for (let b = a + 1; b < cands.length; b++)
      if (!J.seen(cands[a], cands[b])) push(edges, J.judge, cands[a], cands[b]);
}

// ---- metrics --------------------------------------------------------------

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const bestK = mean(truth.slice().sort((a, b) => b - a).slice(0, K));
const allMean = mean(truth);

function score(selected) {
  const trueTop = new Set(Array.from({ length: N }, (_, i) => i)
    .sort((a, b) => truth[b] - truth[a]).slice(0, K));
  return {
    precision: selected.filter((i) => trueTop.has(i)).length / K,
    regret: (bestK - mean(selected.map((i) => truth[i]))) / (bestK - allMean),
  };
}

const STRATS = [
  ["full merge sort",        (J, e, rand) => fullSort(J, e)],
  ["swiss r=2",              (J, e, rand) => swiss(J, e, 2, rand)],
  ["swiss r=4",              (J, e, rand) => swiss(J, e, 4, rand)],
  ["swiss r=6",              (J, e, rand) => swiss(J, e, 6, rand)],
  [`select r=4, ${2}k cands`, (J, e, rand) => selectTopK(J, e, 4, 2, rand)],
  [`select r=4, ${4}k cands`, (J, e, rand) => selectTopK(J, e, 4, 4, rand)],
];

console.log(`n=${N}, top-k=${K}, ${TRIALS} trials, spread=${SPREAD} logits`);
console.log(`adjacent-pair judge accuracy ~${(sigmoid(SPREAD / (N - 1)) * 100).toFixed(1)}%`);
console.log(`objective: pick a k-subset of the strongest items\n`);
console.log("strategy                comparisons   strict precision   normalized regret");
console.log("                                      (exact set match)  (0=optimal, 1=random)");
for (const [label, apply] of STRATS) {
  let cmp = 0, prec = 0, reg = 0;
  for (let t = 0; t < TRIALS; t++) {
    const rand = mulberry(t * 7919 + 13);
    const J = makeJudge(rand), edges = [];
    apply(J, edges, rand);
    const sel = rankOf(fitBT(edges, N)).slice(0, K);
    const s = score(sel);
    cmp += J.count(); prec += s.precision; reg += s.regret;
  }
  console.log(`${label.padEnd(22)} ${(cmp / TRIALS).toFixed(0).padStart(8)}   ` +
    `${(prec / TRIALS * 100).toFixed(1).padStart(13)}%   ${(reg / TRIALS).toFixed(4).padStart(18)}`);
}
