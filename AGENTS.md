# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

`ai-sort` sorts a list of strings by how well each fits a prompt, using an LLM as
a **noisy, non-transitive** pairwise judge. It ships as both a CLI (`ai-sort`) and
a library. Default provider is local **Ollama**; **OpenAI** is also supported, and
any custom `Comparator` works.

Read [README.md](README.md) for the user-facing contract and [ROADMAP.md](ROADMAP.md)
for planned work and the simulation findings behind it — the roadmap records
measurements worth not re-deriving.

## Core design (don't break these invariants)

The pipeline never assumes a well-defined comparison order, because the judge can
report `a > b`, `b > c`, and `c > a`:

1. **Bootstrap** — merge sort drives a first pass (~`n log n` comparisons) to get a
   provisional order. Treated as evidence, not truth.
2. **Refine** — remaining `refinementBudget` goes to the pairs most likely to
   change the order, working outward from score-adjacent neighbours. The
   targeting window widens (capped by `refinementWindow`) and scores refit
   between batches, so any budget is spendable; stops once the budget is spent
   or every pair is settled.
3. **Fit** — all recorded outcomes feed a regularized Bradley-Terry model
   ([bradleyTerry.ts](src/bradleyTerry.ts), Zermelo/MM iterations). Cycles are
   tolerated by construction. Final order sorts by latent score.

Cost/correctness invariants that tests depend on — preserve them:

- **Every pair is judged at most once.** Results are memoized; concurrent requests
  for the same pair share one in-flight call.
- **Identical strings are collapsed** before ranking (duplicates cost nothing, land
  adjacently with equal scores).
- **Presentation order is randomized** per comparison so the judge's position bias
  doesn't masquerade as signal.
- The `Comparator` contract: return `1` if `a` fits better, `-1` if `b` does, `0`
  for a genuine tie ([types.ts](src/types.ts)).

## Layout

| Path | Role |
|---|---|
| [src/index.ts](src/index.ts) | Public library API (`sortByPrompt`, `sortByPromptInPlace`, `rankItemsByPrompt`, `rankByPrompt`, re-exports). |
| [src/sort.ts](src/sort.ts) | The bootstrap + refine pipeline. |
| [src/bradleyTerry.ts](src/bradleyTerry.ts) | MM/Zermelo score fit. |
| [src/cli.ts](src/cli.ts) | CLI entry (`commander`), input/output modes, flags. |
| [src/io.ts](src/io.ts) | File input/output, separators, out-dir handling. |
| [src/types.ts](src/types.ts) | `Comparator`, `SortOptions`, `SortProgress`, `RankedItem`. |
| [src/providers/](src/providers/) | `createComparator` factory, `ollama.ts`, `openai.ts`, `prompt.ts` (judge prompt). |
| [src/sort.test.ts](src/sort.test.ts) | Test suite — uses fake comparators, no LLM calls. |
| [scripts/](scripts/) | Research harnesses (`sim-topk.mjs`, `eval-sst.mjs`, etc.) behind roadmap findings. |
| [examples/](examples/) | `library-usage.mjs`, sample input. |
| [paper/](paper/) | Whitepaper (LaTeX + PDF). |

## Build / test

```bash
npm install
npm run build      # tsc -p tsconfig.json  ->  dist/
npm test           # builds, then runs node --test on dist/**/*.test.js
node examples/library-usage.mjs
```

- `npm test` **builds first**, then runs the compiled tests — there is no separate
  typecheck step; a clean build is the typecheck.
- Tests must stay **LLM-free**: use fake/deterministic comparators (see existing
  ones, including the rock-paper-scissors cycle case). Never add a test that hits
  Ollama or OpenAI.
- Any change touching the pipeline must keep the cycle-handling tests green.

## Conventions

- **TypeScript, strict mode, ESM.** `module`/`moduleResolution` are `NodeNext`, so
  **relative imports use `.js` extensions** even though the sources are `.ts`
  (e.g. `import { rankByPrompt } from "./sort.js"`). Match this.
- Two-space indent, double quotes, semicolons — follow the surrounding code.
- Public API surface lives in [src/index.ts](src/index.ts); export new public types
  and functions there, and keep the JSDoc on them (the README leans on it).
- Only runtime dependency is `commander`. Prefer the standard library over adding
  deps; discuss before introducing a new one.
- Latent scores are on an additive log-odds scale centered near 0, **not**
  probabilities — keep docs/labels consistent with that framing.

## Git workflow (required)

- Commit at **logical units of work** — one coherent change per commit, not a mixed
  dump at the end.
- **Keep everything pushed to `origin`** (https://github.com/omarsaad98/ai-sort).
  Never leave finished work local-only; `git push` after committing.
- Work off `main`. Do not skip hooks or force-push without being asked.
- Build must pass (`npm run build`) and tests must be green (`npm test`) before you
  commit.
