import type { Comparator } from "../types.js";
import { OllamaComparator, type OllamaComparatorOptions } from "./ollama.js";
import { OpenAIComparator, type OpenAIComparatorOptions } from "./openai.js";

export {
  OllamaComparator,
  listOllamaModels,
  resolveOllamaBaseUrl,
  type OllamaComparatorOptions,
} from "./ollama.js";
export { OpenAIComparator, type OpenAIComparatorOptions } from "./openai.js";

export type ComparatorConfig =
  | ({ provider: "ollama" } & OllamaComparatorOptions)
  | ({ provider: "openai" } & OpenAIComparatorOptions);

export function createComparator(config: ComparatorConfig): Comparator {
  switch (config.provider) {
    case "ollama":
      return new OllamaComparator(config);
    case "openai":
      return new OpenAIComparator(config);
    default: {
      const exhaustive: never = config;
      throw new Error(`Unknown provider: ${JSON.stringify(exhaustive)}`);
    }
  }
}
