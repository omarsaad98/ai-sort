/**
 * A comparator decides, given a prompt, which of two items fits the prompt better.
 * Returns 1 if `a` fits better, -1 if `b` fits better, 0 if genuinely tied/undecidable.
 */
export interface Comparator {
  compare(a: string, b: string, prompt: string): Promise<-1 | 0 | 1>;
}

export interface SortProgress {
  /** Number of LLM comparisons made so far. */
  comparisons: number;
  /** Phase of the algorithm. */
  phase: "initial-sort" | "refine";
  /** Total items being sorted. */
  total: number;
}

export interface SortOptions {
  /** The prompt each item is being ranked against. Best fit sorts first. */
  prompt: string;
  /** Comparator implementation. Use createComparator() to build one from provider config. */
  comparator: Comparator;
  /** Max concurrent in-flight comparisons. Default 4. */
  concurrency?: number;
  /**
   * Extra comparison budget spent refining uncertain pairs after the initial
   * O(n log n) merge-sort pass, using Bradley-Terry scores to find the pairs
   * most likely to change the final order. Refinement works outward from the
   * closest neighbours in the current ranking; the targeting window widens once
   * near pairs are exhausted, so any budget up to the full O(n^2) distinct pairs
   * can be spent. Default: n (one extra round's worth), 0 disables refinement
   * entirely. See `refinementWindow` to cap the breadth.
   */
  refinementBudget?: number;
  /**
   * Maximum targeting breadth for refinement: how many ranks apart two items may
   * be and still get compared. Refinement always spends on the closest pairs
   * first; this caps how far the window will widen. Default: unbounded (widens
   * until the budget is spent or every pair has been compared). Set it to keep
   * the extra comparisons focused on near-ties rather than distant pairs.
   */
  refinementWindow?: number;
  /**
   * Strength of the Bradley-Terry regularizing prior — the fictitious wins and
   * losses each item gets against a neutral average opponent. Controls score
   * shrinkage and scale: higher pulls scores toward 0 (tighter, more
   * conservative gaps), lower lets a clean separation stretch the latent scale
   * further. Only affects the reported scores and refinement targeting, never
   * the comparison budget. Must be >= 0. Default: 0.5.
   */
  priorWeight?: number;
  /**
   * Number of minorize-maximize iterations used to fit the Bradley-Terry
   * scores. More iterations converge tighter at more CPU cost (no extra LLM
   * calls); the default is comfortably past convergence for typical sizes.
   * Must be >= 1. Default: 200.
   */
  btIterations?: number;
  /** Called after each comparison completes, useful for progress bars. */
  onProgress?: (progress: SortProgress) => void;
}

export interface RankedItem {
  item: string;
  /**
   * Latent Bradley-Terry quality score, higher fits the prompt better. On the
   * additive scale where P(a beats b) = sigmoid(score_a - score_b), centered
   * near 0 — not a probability. A gap of ~2.2 is roughly 90% confidence; items
   * within a few tenths of each other are effectively tied.
   */
  score: number;
}
