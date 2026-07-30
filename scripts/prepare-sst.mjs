/**
 * Builds a ranking benchmark from the Stanford Sentiment Treebank.
 *
 * SST is a good fit for evaluating ai-sort because its labels are *continuous*
 * (0.0-1.0), not just class buckets. That gives a fine-grained ground-truth
 * total order to measure Kendall tau against, and lets us build test sets of
 * controlled difficulty by choosing how tightly items are spaced.
 *
 * Download first:
 *   curl -sL -o data/sst.zip https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
 *   cd data && unzip -q sst.zip
 *
 * Then:  node scripts/prepare-sst.mjs
 * Writes data/sst-sentences.jsonl  ({ text, score } per line)
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SST = join(ROOT, "data", "stanfordSentimentTreebank");

/** SST ships PTB-tokenized text; undo it so the judge sees natural prose. */
function detokenize(s) {
  return s
    .replace(/-LRB-/g, "(")
    .replace(/-RRB-/g, ")")
    .replace(/\\\//g, "/")
    .replace(/``|''/g, '"')
    .replace(/\s+([,.!?;:%)\]])/g, "$1")
    .replace(/([(\[])\s+/g, "$1")
    .replace(/\s+('s|'re|'ve|'ll|'d|'m|n't)\b/g, "$1")
    .replace(/\s{2,}/g, " ")
    .trim();
}

/** Key used to match a sentence against dictionary.txt, ignoring cosmetic differences. */
function matchKey(s) {
  return s.replace(/\\\//g, "/").replace(/\s+/g, " ").trim().toLowerCase();
}

const scoreOf = new Map(); // phrase id -> continuous sentiment
for (const line of readFileSync(join(SST, "sentiment_labels.txt"), "utf8").split("\n").slice(1)) {
  const [id, value] = line.split("|");
  if (id && value !== undefined) scoreOf.set(id.trim(), Number(value));
}

const idOfPhrase = new Map(); // normalized phrase text -> phrase id
for (const line of readFileSync(join(SST, "dictionary.txt"), "utf8").split("\n")) {
  const sep = line.lastIndexOf("|");
  if (sep <= 0) continue;
  idOfPhrase.set(matchKey(line.slice(0, sep)), line.slice(sep + 1).trim());
}

const rows = [];
let unmatched = 0;
for (const line of readFileSync(join(SST, "datasetSentences.txt"), "utf8").split("\n").slice(1)) {
  const tab = line.indexOf("\t");
  if (tab < 0) continue;
  const raw = line.slice(tab + 1).trim();
  if (!raw) continue;
  const id = idOfPhrase.get(matchKey(raw));
  const score = id !== undefined ? scoreOf.get(id) : undefined;
  if (score === undefined) {
    unmatched++;
    continue;
  }
  rows.push({ text: detokenize(raw), score });
}

const out = join(ROOT, "data", "sst-sentences.jsonl");
writeFileSync(out, rows.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf8");

const scores = rows.map((r) => r.score).sort((a, b) => a - b);
const q = (p) => scores[Math.floor(p * (scores.length - 1))];
const words = rows.map((r) => r.text.split(/\s+/).length).sort((a, b) => a - b);

console.log(`matched   ${rows.length} sentences to continuous scores`);
console.log(`unmatched ${unmatched} (tokenization/encoding drift between the SST files)`);
console.log(`score      min ${q(0).toFixed(3)}  p25 ${q(0.25).toFixed(3)}  median ${q(0.5).toFixed(3)}  p75 ${q(0.75).toFixed(3)}  max ${q(1).toFixed(3)}`);
console.log(`distinct score values: ${new Set(scores).size}`);
console.log(`length     median ${words[Math.floor(words.length / 2)]} words, max ${words[words.length - 1]}`);
console.log(`\nwrote ${out}`);
console.log(`\nmost negative: ${rows.reduce((a, b) => (a.score < b.score ? a : b)).text.slice(0, 90)}`);
console.log(`most positive: ${rows.reduce((a, b) => (a.score > b.score ? a : b)).text.slice(0, 90)}`);
