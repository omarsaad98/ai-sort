// Run from the project root after `npm run build`:
//   node examples/library-usage.mjs
import { sortByPrompt, rankItemsByPrompt, createComparator, listOllamaModels } from "../dist/index.js";

const candidates = [
  "Fix the login button alignment on mobile Safari",
  "Production database is down, all writes are failing",
  "Update the copyright year in the footer",
  "Users report intermittent 500s on checkout, ~3% of traffic",
  "Add a dark mode toggle to settings",
];

const [model] = await listOllamaModels();
const comparator = createComparator({ provider: "ollama", model });

const sorted = await sortByPrompt(candidates, {
  prompt: "the most urgent issue to fix first",
  comparator,
  concurrency: 4,
  onProgress: ({ comparisons }) => process.stderr.write(`\r${comparisons} comparisons`),
});
process.stderr.write("\n");

console.log("Sorted (most urgent first):");
for (const [i, item] of sorted.entries()) console.log(`  ${i + 1}. ${item}`);

// Scores let you spot near-ties instead of trusting an exact rank.
const ranked = await rankItemsByPrompt(candidates, {
  prompt: "the most urgent issue to fix first",
  comparator,
});
console.log("\nWith latent scores:");
for (const { item, score } of ranked) {
  console.log(`  ${score.toFixed(2).padStart(6)}  ${item}`);
}
