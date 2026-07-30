/**
 * Fits Bradley-Terry latent quality scores from pairwise win counts using the
 * Zermelo / minorize-maximize (MM) algorithm. Tolerates cycles and noise in
 * the input by design: it finds the scores that make the *whole* set of
 * observed outcomes most likely, rather than requiring every pair to be
 * satisfied by the final order (which is impossible when a > b > c > a).
 */

export interface PairWins {
  /** wins[i][j] = number of times item i beat item j. */
  wins: Map<number, Map<number, number>>;
}

export function createPairWins(): PairWins {
  return { wins: new Map() };
}

export function recordWin(pw: PairWins, winner: number, loser: number, count = 1): void {
  let row = pw.wins.get(winner);
  if (!row) {
    row = new Map();
    pw.wins.set(winner, row);
  }
  row.set(loser, (row.get(loser) ?? 0) + count);
}

/** Record a tie as half a win for each side, the standard Bradley-Terry treatment. */
export function recordTie(pw: PairWins, a: number, b: number): void {
  recordWin(pw, a, b, 0.5);
  recordWin(pw, b, a, 0.5);
}

function winsBetween(pw: PairWins, i: number, j: number): number {
  return pw.wins.get(i)?.get(j) ?? 0;
}

/**
 * Default strength of the regularizing prior: each item gets this many
 * fictitious wins and losses against a virtual average opponent. Without it, a
 * cleanly separated ordering drives the top and bottom scores toward
 * +/-infinity, so the magnitudes become meaningless (and eventually overflow).
 */
export const DEFAULT_PRIOR_WEIGHT = 0.5;

/** Default cap on MM iterations; the fit normally converges well before this. */
export const DEFAULT_BT_ITERATIONS = 200;

/**
 * Default convergence tolerance: stop once no item's latent (log-scale) score
 * moves more than this between iterations. Tight enough that the early stop is
 * indistinguishable from running the full iteration cap for typical inputs.
 */
export const DEFAULT_BT_TOLERANCE = 1e-8;

/**
 * Fit Bradley-Terry scores for `n` items indexed 0..n-1, on the multiplicative
 * (gamma) scale with geometric mean 1. Items with no recorded comparisons keep
 * the neutral score of 1.
 *
 * `priorWeight` sets the shrinkage of the regularizing prior (higher pulls
 * scores toward the neutral anchor, shrinking the spread; lower lets a clean
 * separation stretch the scale further).
 *
 * The MM iteration runs until it converges — no item's latent (log-scale) score
 * shifts by more than `tolerance` — or until `iterations` iterations have run,
 * whichever comes first. Easy inputs converge in a handful of iterations; the
 * cap only bounds pathological cases. Set `tolerance` to `0` to always run the
 * full cap.
 */
export function fitBradleyTerry(
  pw: PairWins,
  n: number,
  iterations = DEFAULT_BT_ITERATIONS,
  priorWeight = DEFAULT_PRIOR_WEIGHT,
  tolerance = DEFAULT_BT_TOLERANCE,
): number[] {
  const scores = new Array(n).fill(1);
  if (n <= 1) return scores;

  // Precompute the set of opponents each item has faced, so we skip pairs
  // with zero total games (avoids division by zero and wasted work).
  const opponents: Set<number>[] = Array.from({ length: n }, () => new Set());
  for (const [i, row] of pw.wins) {
    for (const [j] of row) {
      if (i === j) continue;
      opponents[i].add(j);
      opponents[j].add(i);
    }
  }

  for (let iter = 0; iter < iterations; iter++) {
    const next = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const opps = opponents[i];
      if (opps.size === 0) {
        next[i] = scores[i];
        continue;
      }
      // Prior: priorWeight wins and priorWeight losses against a virtual
      // opponent of strength 1 (the geometric-mean anchor).
      let numerator = priorWeight;
      let denominator = (2 * priorWeight) / (scores[i] + 1);
      for (const j of opps) {
        const wij = winsBetween(pw, i, j);
        const wji = winsBetween(pw, j, i);
        const total = wij + wji;
        if (total === 0) continue;
        numerator += wij;
        denominator += total / (scores[i] + scores[j]);
      }
      next[i] = denominator > 0 ? numerator / denominator : scores[i];
      if (!Number.isFinite(next[i]) || next[i] <= 0) next[i] = scores[i];
    }
    // Normalize geometric mean to 1 to prevent drift/overflow across iterations.
    const logSum = next.reduce((s, v) => s + Math.log(v > 0 ? v : 1e-9), 0);
    const geoMean = Math.exp(logSum / n);
    for (let i = 0; i < n; i++) next[i] = next[i] / geoMean;

    // Largest per-item move on the latent (log) scale — the quantity the final
    // ordering and scores are read from. Measured after normalization so a
    // uniform rescale doesn't register as progress.
    let maxDelta = 0;
    for (let i = 0; i < n; i++) {
      const delta = Math.abs(Math.log(next[i] / scores[i]));
      if (delta > maxDelta) maxDelta = delta;
    }

    scores.splice(0, n, ...next);

    // Converged: further iterations would not move any score meaningfully.
    if (maxDelta <= tolerance) break;
  }

  return scores;
}

/**
 * Converts gamma-scale scores to the additive latent scale used in
 * P(i beats j) = sigmoid(s_i - s_j). Centered near 0, so a difference of ~2.2
 * means roughly a 90% chance the higher-scored item wins.
 */
export function toLogScale(gammaScores: readonly number[]): number[] {
  return gammaScores.map((g) => Math.log(g > 0 ? g : Number.MIN_VALUE));
}
