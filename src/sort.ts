import type { Comparator, SortOptions, SortProgress } from "./types.js";
import {
  createPairWins,
  recordWin,
  recordTie,
  fitBradleyTerry,
  toLogScale,
  type PairWins,
} from "./bradleyTerry.js";

class Semaphore {
  private queue: Array<() => void> = [];
  private active = 0;
  constructor(private readonly max: number) {}

  async run<T>(fn: () => Promise<T>): Promise<T> {
    if (this.active >= this.max) {
      await new Promise<void>((resolve) => this.queue.push(resolve));
    }
    this.active++;
    try {
      return await fn();
    } finally {
      this.active--;
      const next = this.queue.shift();
      if (next) next();
    }
  }
}

interface Engine {
  values: string[];
  prompt: string;
  comparator: Comparator;
  semaphore: Semaphore;
  cache: Map<string, -1 | 0 | 1>;
  /** In-flight comparisons, keyed the same as `cache`, so a pair is never judged twice. */
  pending: Map<string, Promise<-1 | 0 | 1>>;
  pairWins: PairWins;
  comparisons: number;
  onProgress?: (p: SortProgress) => void;
  total: number;
}

function cacheKey(lo: number, hi: number): string {
  return `${lo}:${hi}`;
}

/**
 * Judges the pair (lo, hi) with lo < hi, returning 1 when `lo` fits the prompt
 * better. Results are memoized, and concurrent requests for the same pair share
 * one in-flight call, so no pair ever costs more than a single LLM request.
 */
function canonicalCompare(
  engine: Engine,
  lo: number,
  hi: number,
  phase: SortProgress["phase"],
): Promise<-1 | 0 | 1> {
  const key = cacheKey(lo, hi);

  const cached = engine.cache.get(key);
  if (cached !== undefined) return Promise.resolve(cached);

  const inFlight = engine.pending.get(key);
  if (inFlight) return inFlight;

  const request = engine.semaphore.run(async () => {
    // Randomize which side is presented first to reduce the model's position bias.
    const flip = Math.random() < 0.5;
    const raw = flip
      ? await engine.comparator.compare(engine.values[hi], engine.values[lo], engine.prompt)
      : await engine.comparator.compare(engine.values[lo], engine.values[hi], engine.prompt);
    // Re-orient so 1 always means "lo fits better", regardless of presentation order.
    const oriented: -1 | 0 | 1 = flip ? (raw === 0 ? 0 : ((-raw) as -1 | 1)) : raw;

    engine.cache.set(key, oriented);
    engine.comparisons++;
    if (oriented === 1) recordWin(engine.pairWins, lo, hi);
    else if (oriented === -1) recordWin(engine.pairWins, hi, lo);
    else recordTie(engine.pairWins, lo, hi);
    engine.onProgress?.({ comparisons: engine.comparisons, phase, total: engine.total });
    return oriented;
  });

  // Registered synchronously, before any await can yield, so a concurrent
  // caller for this pair always finds it here rather than issuing a second call.
  engine.pending.set(key, request);
  return request.finally(() => engine.pending.delete(key));
}

/** Compares two item ids. Returns 1 if `i` fits the prompt better, -1 if `j` does, 0 if tied. */
async function compareIds(
  engine: Engine,
  i: number,
  j: number,
  phase: SortProgress["phase"],
): Promise<-1 | 0 | 1> {
  if (i === j) return 0;
  const lo = Math.min(i, j);
  const hi = Math.max(i, j);
  const canonicalResult = await canonicalCompare(engine, lo, hi, phase);
  // canonicalResult means "lo fits better"; re-orient to mean "i fits better".
  if (canonicalResult === 0) return 0;
  const loWins = canonicalResult === 1;
  const iIsLo = i === lo;
  return (loWins === iIsLo ? 1 : -1) as -1 | 1;
}

async function merge(engine: Engine, left: number[], right: number[]): Promise<number[]> {
  const result: number[] = [];
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    const c = await compareIds(engine, left[i], right[j], "initial-sort");
    if (c >= 0) {
      result.push(left[i]);
      i++;
    } else {
      result.push(right[j]);
      j++;
    }
  }
  while (i < left.length) result.push(left[i++]);
  while (j < right.length) result.push(right[j++]);
  return result;
}

async function mergeSort(engine: Engine, ids: number[]): Promise<number[]> {
  if (ids.length <= 1) return ids;
  const mid = ids.length >> 1;
  const [left, right] = await Promise.all([
    mergeSort(engine, ids.slice(0, mid)),
    mergeSort(engine, ids.slice(mid)),
  ]);
  return merge(engine, left, right);
}

/**
 * Spends up to `budget` extra comparisons on the pairs most likely to change
 * the final order, working outward from the closest neighbours in the current
 * score ranking. Refits scores after each batch so later batches target the
 * updated ranking.
 *
 * The targeting window starts at distance 1 (immediate neighbours) and widens
 * one rank at a time, but only once every uncompared pair inside it is settled
 * — so the budget always lands on the nearest, most order-relevant pairs first.
 * This is what lets an arbitrarily large budget be spent: a fixed w=1 pass caps
 * out at ~n comparisons, whereas widening exposes the O(n^2) pairs further
 * apart. `maxWindow` bounds how far the window may grow (at n-1 every pair has
 * been compared).
 */
async function refine(
  engine: Engine,
  n: number,
  budget: number,
  batchSize: number,
  maxWindow: number,
  iterations: number,
  priorWeight: number,
  tolerance: number,
): Promise<void> {
  let spent = 0;
  let window = 1;
  while (spent < budget) {
    const scores = fitBradleyTerry(engine.pairWins, n, iterations, priorWeight, tolerance);
    const order = Array.from({ length: n }, (_, id) => id).sort((a, b) => scores[b] - scores[a]);

    // Collect a batch of uncompared pairs within the current window, nearest
    // neighbours first: distance 1, then 2, and so on up to `window`. A refit
    // may have reordered items, so re-scanning from distance 1 each round
    // catches pairs that only just became adjacent. Cached pairs are settled —
    // re-querying cannot change a memoized verdict — so they are skipped.
    // Batching lets the LLM calls overlap instead of running one at a time.
    const batch: Array<[number, number]> = [];
    const room = Math.min(batchSize, budget - spent);
    for (let d = 1; d <= window && batch.length < room; d++) {
      for (let k = 0; k + d < order.length && batch.length < room; k++) {
        const a = order[k];
        const b = order[k + d];
        if (!engine.cache.has(cacheKey(Math.min(a, b), Math.max(a, b)))) {
          batch.push([a, b]);
        }
      }
    }

    if (batch.length === 0) {
      // Every pair within the current window is settled. Widen it if allowed
      // and farther pairs remain; otherwise nothing is left to learn.
      if (window >= maxWindow || window >= n - 1) break;
      window++;
      continue;
    }

    await Promise.all(batch.map(([a, b]) => compareIds(engine, a, b, "refine")));
    spent += batch.length;
  }
}

export interface RankResult {
  /** Item ids (0-indexed into the original input array), best fit first. */
  order: number[];
  /**
   * Latent Bradley-Terry score per id, aligned with the original input array,
   * on the additive scale where P(i beats j) = sigmoid(scores[i] - scores[j]).
   * Centered near 0. Items whose scores are close are effectively tied.
   */
  scores: number[];
  comparisons: number;
}

export async function rankByPrompt(items: readonly string[], options: SortOptions): Promise<RankResult> {
  const n = items.length;
  if (n <= 1) {
    return { order: n === 1 ? [0] : [], scores: n === 1 ? [0] : [], comparisons: 0 };
  }

  // Collapse identical texts into one group before ranking. Two identical
  // strings can only ever tie, so judging them — or judging each duplicate
  // against the same opponent — would burn LLM calls for no information.
  const groupOfItem = new Array<number>(n);
  const groupTexts: string[] = [];
  const itemsOfGroup: number[][] = [];
  const textToGroup = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    let group = textToGroup.get(items[i]);
    if (group === undefined) {
      group = groupTexts.length;
      textToGroup.set(items[i], group);
      groupTexts.push(items[i]);
      itemsOfGroup.push([]);
    }
    groupOfItem[i] = group;
    itemsOfGroup[group].push(i);
  }

  const groupCount = groupTexts.length;
  const engine: Engine = {
    values: groupTexts,
    prompt: options.prompt,
    comparator: options.comparator,
    semaphore: new Semaphore(Math.max(1, options.concurrency ?? 4)),
    cache: new Map(),
    pending: new Map(),
    pairWins: createPairWins(),
    comparisons: 0,
    onProgress: options.onProgress,
    total: groupCount,
  };

  // Bradley-Terry fit dials. Both only affect scoring/refinement targeting, not
  // the LLM comparison budget. Clamp to safe ranges so a bad option can't
  // diverge the fit or spin forever.
  const btIterations = Math.max(1, Math.floor(options.btIterations ?? 200));
  const priorWeight = Math.max(0, options.priorWeight ?? 0.5);
  const btTolerance = Math.max(0, options.btTolerance ?? 1e-8);

  let groupScores: number[];
  if (groupCount <= 1) {
    groupScores = new Array(groupCount).fill(0);
  } else {
    const ids = Array.from({ length: groupCount }, (_, i) => i);
    await mergeSort(engine, ids);

    const budget = options.refinementBudget ?? groupCount;
    if (budget > 0) {
      // Unbounded window by default: widen until the budget is spent or every
      // pair (distance up to n-1) has been compared.
      const maxWindow = Math.max(1, options.refinementWindow ?? groupCount - 1);
      await refine(
        engine,
        groupCount,
        budget,
        Math.max(1, options.concurrency ?? 4),
        maxWindow,
        btIterations,
        priorWeight,
        btTolerance,
      );
    }
    groupScores = toLogScale(
      fitBradleyTerry(engine.pairWins, groupCount, btIterations, priorWeight, btTolerance),
    );
  }

  const groupOrder = Array.from({ length: groupCount }, (_, g) => g).sort(
    (a, b) => groupScores[b] - groupScores[a],
  );

  // Expand groups back to original item ids; duplicates land adjacently.
  const order = groupOrder.flatMap((g) => itemsOfGroup[g]);
  const scores = Array.from({ length: n }, (_, i) => groupScores[groupOfItem[i]]);

  return { order, scores, comparisons: engine.comparisons };
}
