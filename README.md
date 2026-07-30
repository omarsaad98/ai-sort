# ai-sort

Sort a list of strings by how well each one fits a prompt, using an LLM as the pairwise judge. Works with a local **Ollama** install (default) or **OpenAI**. Ships as both a CLI and a library.

```bash
$ ai-sort --input snacks.txt --prompt "the healthiest snack choice" --scores
[ 1.87] Plain Greek yogurt with fresh blueberries and chia seeds
[ 1.61] Raw almonds and walnuts, unsalted
[ 1.34] A bowl of steamed broccoli with lemon and olive oil
[ 0.42] Grilled salmon fillet with a side of quinoa
[-0.23] Neon-orange cheese puffs from a gas station
[-0.96] A king-size milk chocolate bar with caramel filling
[-1.91] Deep-fried funnel cake buried in powdered sugar
[-2.15] A large glazed doughnut with rainbow sprinkles
```

## Install

```bash
npm install ai-sort        # library
npm install -g ai-sort     # CLI
```

Requires Node 18.17+. For the default provider, have [Ollama](https://ollama.com) running locally with at least one model pulled.

## How it works

An LLM asked "which of these two fits better?" is a **noisy, non-transitive** comparator: it can report a > b, b > c, and c > a. No ordinary comparison sort is well-defined on that, so ai-sort does not rely on one:

1. **Bootstrap** — a merge sort drives the first pass, using roughly `n log n` comparisons to get a provisional order. Its output is treated as evidence, not truth.
2. **Refine** — remaining budget goes to adjacent pairs that were never directly compared, i.e. the ones most likely to change the result. Batched to run concurrently, and it stops early once every adjacent pair is settled.
3. **Fit** — all recorded outcomes feed a regularized [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) model (Zermelo/MM iterations), which finds the latent scores making the *whole set* of observed judgements most likely. Cycles are tolerated by construction rather than discarded. The final order sorts by that score.

Cost control, since LLM calls are the expensive part:

- Every pair is judged **at most once** — results are memoized, and concurrent requests for the same pair share one in-flight call.
- **Identical strings are collapsed** before ranking, so duplicates cost nothing and land adjacently with equal scores.
- Each comparison **randomizes which candidate is presented first**, so the model's position bias doesn't masquerade as signal.

### Reading the scores

Scores are on the additive latent scale where `P(a beats b) = sigmoid(score_a - score_b)`, centered near 0. A gap of ~2.2 is about 90% confidence; items within a few tenths are **effectively tied** and their relative order is not meaningful. Prefer tiers over exact ranks when the gaps are small — with a near-tie at the top, two runs can legitimately disagree.

## CLI

### Input — pick one

```bash
# One file, split on a separator (default: newline)
ai-sort --input items.txt --prompt "..."
ai-sort --input items.txt --separator '\n---\n' --prompt "..."

# One file per item
ai-sort a.txt b.txt c.txt --prompt "..."
ai-sort ./candidates/*.txt --prompt "..."
```

### Output — default is stdout

```bash
# Single file, items joined by the separator
ai-sort --input items.txt --prompt "..." --output ranked.txt --output-separator '\n---\n'

# One numbered file per item, so a lexical listing matches the ranking
ai-sort ./items/*.txt --prompt "..." --out-dir ./ranked            # 1.txt, 2.txt, ...
ai-sort ./items/*.txt --prompt "..." --out-dir ./ranked --keep-names  # 1-best.txt, 2-next.txt, ...

# Machine-readable
ai-sort --input items.txt --prompt "..." --json
```

Existing output files and non-empty output directories are refused unless you pass `--force`.

### Providers

```bash
# Ollama (default) — defaults to your most recently pulled model
ai-sort --input items.txt --prompt "..." --model qwen3.5:9b
ai-sort --input items.txt --prompt "..." --base-url http://otherhost:11434

# OpenAI — reads $OPENAI_API_KEY (or a .env file in the working directory)
ai-sort --input items.txt --prompt "..." --provider openai --model gpt-5.4-mini
```

For OpenAI, the key is read from the `OPENAI_API_KEY` environment variable. A
`.env` file in the current directory is loaded automatically (real environment
variables take precedence), or pass `--api-key` explicitly. Transient rate
limits and 5xx responses are retried with exponential backoff, honoring any
`Retry-After` header. Point `--base-url` at any OpenAI-compatible endpoint.

Reasoning is disabled by default on thinking-capable Ollama models, since it roughly triples latency for a binary judgement. Pass `--think` to enable it.

### Useful flags

| Flag | Effect |
|---|---|
| `-c, --concurrency <n>` | Max concurrent LLM calls (default 4) |
| `-r, --refine <n>` | Extra comparisons after the first pass (default: item count; `0` disables) |
| `--refine-window <n>` | Cap how many ranks apart refined pairs may be (default: unbounded) |
| `--prior-weight <w>` | Bradley-Terry prior strength; higher shrinks score gaps (default: 0.5) |
| `--bt-iterations <n>` | Bradley-Terry fit iterations; more converges tighter, no extra LLM calls (default: 200) |
| `--scores` | Prefix each line with its latent score |
| `--json` | Emit ranks, scores, and source paths as JSON |
| `--no-trim` | Keep surrounding whitespace on each item |
| `-q, --quiet` | Silence progress on stderr |

Run `ai-sort --help` for the full list.

## Library

```ts
import { sortByPrompt, sortByPromptInPlace, rankItemsByPrompt, createComparator } from "ai-sort";

const comparator = createComparator({ provider: "ollama", model: "qwen3.5:9b" });
// or: createComparator({ provider: "openai", model: "gpt-5.4-mini" })  // uses $OPENAI_API_KEY

// Returns a new sorted array; the input is untouched.
const sorted = await sortByPrompt(items, {
  prompt: "the most urgent issue to fix first",
  comparator,
  concurrency: 4,
  refinementBudget: items.length,
  onProgress: ({ comparisons, phase }) => console.error(`${phase}: ${comparisons}`),
});

// Sorts the given array in place and returns it.
await sortByPromptInPlace(items, { prompt: "...", comparator });

// Same ordering, plus each item's latent score, for tiering near-ties.
const ranked = await rankItemsByPrompt(items, { prompt: "...", comparator });
// -> [{ item: "...", score: 1.87 }, ...]
```

`rankByPrompt` is the lower-level entry point, returning `{ order, scores, comparisons }` where `order` holds indices into the original array and `scores` is aligned to it.

### Custom comparators

Any judge implementing the `Comparator` interface works — another provider, a heuristic, a human in the loop:

```ts
import type { Comparator } from "ai-sort";

const byLength: Comparator = {
  async compare(a, b, prompt) {
    return a.length === b.length ? 0 : a.length > b.length ? 1 : -1;
  },
};
```

Return `1` if `a` fits better, `-1` if `b` does, `0` for a genuine tie.

## Development

```bash
npm install
npm run build
npm test           # builds, then runs the suite (no LLM calls; uses fake comparators)
node examples/library-usage.mjs
```

The test suite covers the cycle handling explicitly, including a rock-paper-scissors comparator and a cycle embedded in an otherwise transitive list.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned dials and the measurements behind them — comparison-density vs accuracy curves, confidence-signal comparisons, and the tradeoffs behind the `refinementBudget`/`refinementWindow` refinement dials.

## License

MIT
