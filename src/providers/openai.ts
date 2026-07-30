import type { Comparator } from "../types.js";
import { COMPARE_SYSTEM_PROMPT, buildCompareUserMessage, parseWinner } from "./prompt.js";

export interface OpenAIComparatorOptions {
  model: string;
  /** Defaults to the OPENAI_API_KEY env var. */
  apiKey?: string;
  /** Defaults to https://api.openai.com/v1, override for compatible endpoints. */
  baseUrl?: string;
}

export class OpenAIComparator implements Comparator {
  private readonly model: string;
  private readonly apiKey: string;
  private readonly baseUrl: string;

  constructor(options: OpenAIComparatorOptions) {
    this.model = options.model;
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OpenAI API key not provided. Pass apiKey or set OPENAI_API_KEY.");
    }
    this.apiKey = apiKey;
    this.baseUrl = (options.baseUrl ?? "https://api.openai.com/v1").replace(/\/+$/, "");
  }

  async compare(a: string, b: string, prompt: string): Promise<-1 | 0 | 1> {
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: COMPARE_SYSTEM_PROMPT },
          { role: "user", content: buildCompareUserMessage(prompt, a, b) },
        ],
      }),
    });

    if (!res.ok) {
      throw new Error(`OpenAI request failed (${res.status}): ${await res.text()}`);
    }

    const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
    const content = data.choices?.[0]?.message?.content ?? "";
    return parseWinner(content);
  }
}
