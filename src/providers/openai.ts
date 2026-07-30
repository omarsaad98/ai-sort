import type { Comparator } from "../types.js";
import { COMPARE_SYSTEM_PROMPT, buildCompareUserMessage, parseWinner } from "./prompt.js";

export interface OpenAIComparatorOptions {
  model: string;
  /** Defaults to the OPENAI_API_KEY env var. */
  apiKey?: string;
  /** Defaults to https://api.openai.com/v1, override for compatible endpoints. */
  baseUrl?: string;
  /** Max attempts per comparison on transient errors (rate limits, 5xx). Default 4. */
  maxRetries?: number;
}

/** Transient HTTP statuses worth retrying with backoff. */
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

export class OpenAIComparator implements Comparator {
  private readonly model: string;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly maxRetries: number;

  constructor(options: OpenAIComparatorOptions) {
    this.model = options.model;
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OpenAI API key not provided. Set OPENAI_API_KEY (a .env file works) or pass --api-key.",
      );
    }
    this.apiKey = apiKey;
    this.baseUrl = (options.baseUrl ?? "https://api.openai.com/v1").replace(/\/+$/, "");
    this.maxRetries = options.maxRetries ?? 4;
  }

  async compare(a: string, b: string, prompt: string): Promise<-1 | 0 | 1> {
    const body = JSON.stringify({
      model: this.model,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: COMPARE_SYSTEM_PROMPT },
        { role: "user", content: buildCompareUserMessage(prompt, a, b) },
      ],
    });

    for (let attempt = 0; ; attempt++) {
      let res: Response;
      try {
        res = await fetch(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${this.apiKey}`,
          },
          body,
        });
      } catch (cause) {
        // Network-level failure (DNS, connection reset). Retry a few times.
        if (attempt < this.maxRetries) {
          await sleep(backoffMs(attempt));
          continue;
        }
        throw new Error(`OpenAI request failed (network error): ${(cause as Error).message}`);
      }

      if (res.ok) {
        const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
        return parseWinner(data.choices?.[0]?.message?.content ?? "");
      }

      const text = await res.text();

      // Out of credit/quota is not transient — retrying just burns time.
      if (res.status === 429 && /insufficient_quota/.test(text)) {
        throw new Error(
          "OpenAI request rejected: your account is out of quota/credit (insufficient_quota). " +
            "Add billing at https://platform.openai.com/account/billing, then retry.",
        );
      }
      if (res.status === 401) {
        throw new Error("OpenAI request rejected (401): the API key is invalid or revoked.");
      }

      if (RETRYABLE_STATUS.has(res.status) && attempt < this.maxRetries) {
        await sleep(retryAfterMs(res) ?? backoffMs(attempt));
        continue;
      }

      throw new Error(`OpenAI request failed (${res.status}): ${text}`);
    }
  }
}

/** Exponential backoff with jitter: ~0.5s, 1s, 2s, 4s ... capped at 20s. */
function backoffMs(attempt: number): number {
  const base = Math.min(500 * 2 ** attempt, 20_000);
  return base + Math.floor(Math.random() * 250);
}

/** Honors a `Retry-After` header (seconds or HTTP-date) when the server sends one. */
function retryAfterMs(res: Response): number | undefined {
  const header = res.headers.get("retry-after");
  if (!header) return undefined;
  const seconds = Number(header);
  if (Number.isFinite(seconds)) return seconds * 1000;
  const date = Date.parse(header);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
