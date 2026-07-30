import { rankByPrompt } from "./sort.js";
import type { RankedItem, SortOptions } from "./types.js";

export type { Comparator, SortOptions, SortProgress, RankedItem } from "./types.js";
export {
  createComparator,
  OllamaComparator,
  OpenAIComparator,
  listOllamaModels,
} from "./providers/index.js";
export type { ComparatorConfig, OllamaComparatorOptions, OpenAIComparatorOptions } from "./providers/index.js";
export { rankByPrompt } from "./sort.js";
export type { RankResult } from "./sort.js";

/**
 * Sorts a copy of `items` so the entry that best fits `options.prompt` comes
 * first. Uses an LLM pairwise comparator with an adaptive merge-sort +
 * Bradley-Terry scoring pipeline, which stays well-behaved even when some
 * comparisons are non-transitive (a > b > c > a).
 */
export async function sortByPrompt(items: readonly string[], options: SortOptions): Promise<string[]> {
  const { order } = await rankByPrompt(items, options);
  return order.map((id) => items[id]);
}

/** Same as sortByPrompt, but sorts the given array in place and returns it. */
export async function sortByPromptInPlace(items: string[], options: SortOptions): Promise<string[]> {
  const { order } = await rankByPrompt(items, options);
  const copy = order.map((id) => items[id]);
  for (let i = 0; i < items.length; i++) items[i] = copy[i];
  return items;
}

/**
 * Like sortByPrompt, but also returns each item's latent Bradley-Terry
 * quality score (higher fits the prompt better), useful for tiering
 * near-tied items instead of trusting an exact rank.
 */
export async function rankItemsByPrompt(items: readonly string[], options: SortOptions): Promise<RankedItem[]> {
  const { order, scores } = await rankByPrompt(items, options);
  return order.map((id) => ({ item: items[id], score: scores[id] }));
}
