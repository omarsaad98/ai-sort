# Whitepaper draft

Research notes on rank inference from a non-transitive pairwise oracle. Scope is
deliberately restricted to the mathematics: problem formulation, estimator
properties, query complexity and span, and empirical evaluation. Implementation
and operational concerns belong in [ROADMAP.md](../ROADMAP.md), not here.

## Build

```bash
cd paper
pdflatex -interaction=nonstopmode whitepaper.tex   # run twice to resolve refs
```

Output: `whitepaper.pdf`, 9 pages. The PDF is gitignored; only the source is tracked.

## Structure

| Section | Content |
|---|---|
| §2 | Query model (single-query-per-pair ⇒ deterministic oracle), objectives |
| §3.1 | Regularized Bradley-Terry estimator, Eq. (5) |
| §3.3 | Proposition 1: merge-sort span is Θ(n); round-based schedules are O(r) |
| §4.2 | Query density vs accuracy |
| §4.3 | Ranking vs selection objective |
| §4.4 | Uncertainty estimators |
| §4.5 | Stanford Sentiment Treebank evaluation |

## Reproducing the numbers

| Table | Source |
|---|---|
| Tab. 1 (density) | not yet committed — **open item** |
| Tab. 2 (selection) | `scripts/sim-topk.mjs` |
| Tab. 3 (uncertainty) | not yet committed — **open item** |
| Tab. 4, 5 (SST) | `scripts/prepare-sst.mjs`, `scripts/eval-sst.mjs` |
| Tab. 6 (SST regret) | `scripts/analyze-dump.mjs data/rank-500.json` |

Estimator properties asserted in §3.1 (cycle absorption, opponent weighting,
divergence without regularization) are exercised by `src/sort.test.ts`.

## Status

Draft. Known gaps, in order of how much they'd change the conclusions:

1. **Citations were written from memory and need verification.** Volume and page
   numbers on several entries are unconfirmed. Do not circulate before checking.
2. Two of six tables are not reproducible from committed code (see above).
3. Single oracle, one sampled instance per size. The n=500 instance was
   replicated once (τ_b 0.6127 vs 0.6096), which bounds presentation-order
   variance only — not variance across item samples, criteria, or models.
4. The synthetic oracle is drawn from the same parametric family used for
   inference, so no misspecification is probed.
