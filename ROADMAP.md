# Roadmap / future work

The goal is a **dial-heavy** tool: sorting with a noisy judge is a cost/accuracy tradeoff, and almost every constant in the pipeline is a knob someone will want to turn. Anything hardcoded below is a dial that hasn't been exposed yet.

## Tunable dials

| Dial | Today | Wanted |
|---|---|---|
| `objective` | always a full total order | `rank` \| `topk` \| `tiers` — a full sort is the *wrong algorithm* for selection (see below) |
| `concurrency` | exposed, default 4 | keep; add adaptive backoff on provider 429/503 |
| `refinementBudget` | exposed, **but caps out at ~n** | expanding window (see below) so any budget is spendable |
| `refinementWindow` | — | targeting breadth `w`; grow until budget is spent |
| `priorWeight` | hardcoded `0.5` (`bradleyTerry.ts`) | expose; controls score shrinkage and scale |
| `btIterations` | hardcoded `200` | expose, or converge on delta tolerance instead |
| `confidenceSource` | none | `none` \| `logprob` \| `verbal` |
| `samplesPerPair` | effectively 1 (memoized) | `k` samples for a real per-pair win rate `p_ij` |
| `softWeights` | off (binary edges) | feed confidence as fractional win weight; **keep default off until calibrated** |
| `positionBias` | randomized order | add `bothOrders` mode (2x cost, cancels bias empirically) |
| `bootstrap` | merge sort | `mergesort` \| `sparse-graph` \| `knockout` \| `network`; merge sort's critical path is Θ(n) serial calls (see below) |
| `scorer` | Bradley-Terry (MM) | add Rank Centrality, approximate weighted Kemeny |
| `tierThreshold` | — | emit tiers instead of exact ranks when score gaps are small |
| `topK` | — | spend budget only near the rank-`k` boundary |
| `stopWhen` | fixed budget | stability-based: stop when top-`k` is stable across refits |
| `judgePrompt` | fixed system prompt | override + few-shot examples |

## Findings that motivate these

Measured during initial development — worth not re-deriving.

### Comparison density pays, with sharp diminishing returns

Simulation, n=60, 24 trials, ground-truth known, judge ~51.7% accurate on adjacent pairs. `tau-dist` = fraction of pairs misordered.

| strategy | comparisons | tau-dist | top-10 |
|---|---|---|---|
| merge sort only | 227 | 0.2293 | 5.5/10 |
| window w=1 | 328 | 0.1680 | 6.0/10 |
| window w=2 | 412 | 0.1464 | 6.8/10 |
| window w=4 | 558 | 0.1224 | 7.0/10 |
| window w=8 | 784 | 0.1044 | 7.4/10 |
| exhaustive | ~1997 | 0.0792 | 8.3/10 |

- Error drops ~65% from `n log n` to exhaustive. The first ~100 extra comparisons are the best value (−27% error for +44% cost).
- **Targeted beats random by only ~8%** at equal budget, consistently. Raw density is most of the win; sophisticated pair selection is a refinement, not the main event. Don't over-invest there first.
- Extra edges help even though each pair is judged once and memoized: a wrong edge can't be resampled, but it gets *outvoted* as an item accumulates opponents.
- **Hard noise floor.** Exhaustive comparison still leaves tau at 0.079. Judge quality caps achievable accuracy — past that, only a better judge or honest tiers help.
- Caveat: the exhaustive row double-counts merge-sort pairs, so its count exceeds the 1770 unique pairs and those edges carry double weight. One noise model, one graph size — trust the curve's shape, not the exact figures.

### Sorting is the wrong algorithm if you only need a good subset

Often the user wants *some subset of the best* — 10 good candidates — and doesn't
care about the order within it, or which of two near-identical items got picked.
That's a different objective with a different cost.

Two ways to score a selected k-subset:

- **strict precision** — exact set overlap with the true top-k. Treats swapping
  two near-identical items as a total miss.
- **normalized regret** — `(best − got) / (best − random)` on mean true quality.
  `0` = optimal, `1` = no better than random. Near-tie confusions cost ~nothing.

Simulation (`scripts/sim-topk.mjs`), n=200, k=10, 20 trials, judge 50.5% accurate
on adjacent pairs. *Swiss* = r rounds of pairings (later rounds pair similar
strengths); *select* = Swiss screening then all-pairs among the top `ck`.

| strategy | comparisons | strict precision | normalized regret |
|---|---|---|---|
| full merge sort | 1042 | 24.0% | 0.3740 |
| swiss r=2 | 200 | 6.5% | 0.7298 |
| swiss r=4 | 400 | 15.5% | 0.3622 |
| swiss r=6 | 598 | 23.5% | 0.2556 |
| **select r=4, 2k cands** | **582** | **26.0%** | **0.1866** |
| select r=4, 4k cands | 1152 | 42.0% | 0.1012 |

- **The metric matters more than the algorithm.** Merge sort's 24% strict
  precision looks bad, but regret 0.374 means the subset captures ~63% of the
  available quality edge over a random pick. Set-overlap understates usefulness
  whenever the top of the list is tightly packed — which is exactly when the
  judge is unreliable.
- **Selection dominates sorting here.** Select/2k gets *half* the regret of a
  full sort at 56% of the cost, and wins on strict precision too. Swiss r=4
  matches merge sort's regret at 38% of the cost. A full sort wastes budget
  ordering items that get discarded, and ordering retained items nobody asked to
  have ordered.
- **It's also far more parallel.** Swiss rounds pair disjoint items, so all ~n/2
  comparisons per round are independent — critical path is O(rounds), not Θ(n).
  For n=200 at 374ms and concurrency 8: ~19s floor vs ~150s for merge sort. This
  fixes the cost *and* the latency problem at once.
- Caveat: synthetic judge, one noise model. The SST run's regret is still
  unmeasured (needs the ranking dump).

### Confidence: prefer logprobs over self-report

Tested on Ollama 0.32.5 / qwen3.5:9b, 3 pairs of known difficulty, both presentation orders.

| pair | verbalized | logprob |
|---|---|---|
| obvious | `1.00 / 1.00`, stable | `0.999 / 0.998`, stable |
| ambiguous | `0.60 / 0.50`, flipped | `0.608 / 0.642`, stable |
| near-tie | `0.60 / 0.60`, stable | `0.525 / 0.825`, flipped |

- Both signals separate easy (~1.0) from hard (0.5–0.65) — the property needed for active sampling.
- Verbalized confidence is **badly quantized**: 6 calls produced 3 distinct values, clustering on round numbers. Logprobs are continuous and finer-grained.
- **Logprobs are free**; a `confidence` field costs output tokens on every call, and generation is the dominant cost.
- Neither is order-invariant — each flipped on one hard pair. The most trustworthy confidence is empirical agreement across both orders (2x cost, also cancels position bias).
- Only 3 pairs, one model. This shows the signal discriminates difficulty; it does **not** establish that `0.6` means 60%. Real calibration needs a labeled set and a reliability curve.
- Requires the verdict to be the first token, so `format: "json"` must be dropped in favor of a bare `A`/`B` (`parseWinner` already has a loose-text fallback). Verified on Ollama; **not** yet verified on OpenAI, and reasoning endpoints may not expose logprobs.

## Bigger items

- **Expanding-window refine** — the highest-value change. `refine` currently stops once every score-adjacent pair is compared (w=1), so a large `refinementBudget` goes unspent; the snacks run had budget 8 and used 2. Grow `w` and refit between rounds until the budget is exhausted, turning `refinementBudget` into a real dial across the whole table above.
- **Persistent comparison graph** — cache the win graph to disk keyed by (item text, prompt, model). Makes re-runs nearly free and enables incremental work.
- **Incremental insert** — add items to an existing ranking without re-sorting; binary-search the ranking, then refine locally.
- **Tiers as a first-class output** — group statistically indistinguishable items instead of asserting rank 17 vs 18. Needs score uncertainty (bootstrap over the edge set, or Hessian-based intervals).
- **Batch judging** — several pairs per request to amortize latency. Risks cross-contamination between judgements; measure before adopting.
- **Top-k mode** — most budget near the cutoff. Large saver when you need the top 20 of 100 and don't care about ordering the bottom.
- **Token/cost accounting** — report tokens and estimated spend; support a hard cost ceiling.
- **Providers** — Anthropic; llama.cpp direct. OpenAI-compatible endpoints already work via `--base-url`.
- **Calibration harness** — a labeled benchmark set to actually measure judge accuracy and confidence calibration per model, so these dials can be tuned on evidence instead of intuition.

## Performance notes

Measured on an RTX 3070 (8 GiB), Ollama 0.32.5, qwen3.5:9b Q4_K_M, 4096 context.

| variant | `ollama ps` PROCESSOR | resident | warm latency | throughput |
|---|---|---|---|---|
| `qwen3.5:9b` | `12%/88% CPU/GPU` | 6.3 GB | 424 ms | 62.4 tok/s |
| `qwen3.5:9b-gpu` (`num_gpu 999`) | `100% GPU` | 5.6 GB | 374 ms | 75.7 tok/s |

- Forcing full GPU residency is worth **~12% latency / ~21% throughput** here. Real, but not transformative — 8 GiB is simply tight for a 6.3 GB model, and the 12% CPU spill was only costing ~50 ms/call. Prefer the `-gpu` tags; check `PROCESSOR` reads `100% GPU`.
- **Warm per-call latency is ~370–420 ms, not seconds.** An early 69s figure for an 8-item sort was a one-off first-ever model load (reading 6.6 GB from disk before the OS page cache was warm). Reproducible numbers for that same sort: 5.4s warm, 8.9s cold. Never infer per-call latency by dividing wall time by comparison count.
- Model load costs ~4.4s once and is identical for both variants; keep the model warm across a run.
- Reasoning is disabled by default (`think: false`) since it roughly triples latency for a binary verdict.

### Concurrency is bounded by the scheduler, not the limit

For the 8-item sort: `-c 1` took 10.7s vs `-c 4` at 8.9s — concurrency 4 bought only ~17%.

Merge sort's merges are inherently sequential, and the **final merge of the two halves alone is up to `n-1` strictly serial comparisons**. So the critical path is Θ(n) no matter how high `concurrency` goes. At ~400 ms/call, a 100-item sort pays ≥40s in the root merge alone, with most workers idle.

This makes the scheduler a priority dial:

- **Parallel bootstrap instead of merge sort.** The merge sort is *only* a query scheduler — the actual ordering comes from the Bradley-Terry fit. A structured sparse graph (each item vs `w` others) can be issued at depth 1, fully parallel, then fitted. The density experiment found random extra edges only ~8% worse than targeted ones at equal budget, which suggests a parallel random/structured graph could match merge-sort quality at a fraction of the wall time. Needs testing: that experiment measured random edges *added to* a merge-sort base, not as a replacement for it.
- **Sorting networks** (bitonic/odd-even) give a fixed comparison pattern with O(log²n) depth and wide per-round parallelism.
- **Knockout rounds** are fully parallel per round, but yield a max rather than a full order — useful for `topK`.
