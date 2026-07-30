import type { Comparator } from "../types.js";
import { COMPARE_SYSTEM_PROMPT, buildCompareUserMessage, parseWinner } from "./prompt.js";

export interface OllamaComparatorOptions {
  model: string;
  /** Defaults to http://localhost:11434 or OLLAMA_HOST env var. */
  baseUrl?: string;
  /**
   * Disable chain-of-thought on thinking-capable models. Reasoning roughly
   * triples latency for a binary judgement, so it is off by default; pass
   * true if you want the model to reason before answering.
   */
  think?: boolean;
}

export function resolveOllamaBaseUrl(baseUrl?: string): string {
  const raw = baseUrl ?? process.env.OLLAMA_HOST ?? "http://localhost:11434";
  const withScheme = /^https?:\/\//.test(raw) ? raw : `http://${raw}`;
  return withScheme.replace(/\/+$/, "");
}

/** Lists locally installed Ollama models, most recently modified first. */
export async function listOllamaModels(baseUrl?: string): Promise<string[]> {
  const url = `${resolveOllamaBaseUrl(baseUrl)}/api/tags`;
  let res: Response;
  try {
    res = await fetch(url);
  } catch (cause) {
    throw new Error(
      `Cannot reach Ollama at ${resolveOllamaBaseUrl(baseUrl)}. Is it running? (${String(cause)})`,
    );
  }
  if (!res.ok) {
    throw new Error(`Ollama /api/tags failed (${res.status}): ${await res.text()}`);
  }
  const data = (await res.json()) as { models?: Array<{ name?: string; modified_at?: string }> };
  return (data.models ?? [])
    .slice()
    .sort((a, b) => String(b.modified_at ?? "").localeCompare(String(a.modified_at ?? "")))
    .map((m) => m.name)
    .filter((name): name is string => Boolean(name));
}

export class OllamaComparator implements Comparator {
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly think: boolean;
  /**
   * Models without thinking support reject the `think` field outright, so we
   * drop it permanently the first time Ollama complains about it.
   */
  private sendThink = true;

  constructor(options: OllamaComparatorOptions) {
    this.model = options.model;
    this.baseUrl = resolveOllamaBaseUrl(options.baseUrl);
    this.think = options.think ?? false;
  }

  private async request(includeThink: boolean, a: string, b: string, prompt: string): Promise<Response> {
    return fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        stream: false,
        format: "json",
        ...(includeThink ? { think: this.think } : {}),
        options: { temperature: 0 },
        messages: [
          { role: "system", content: COMPARE_SYSTEM_PROMPT },
          { role: "user", content: buildCompareUserMessage(prompt, a, b) },
        ],
      }),
    });
  }

  async compare(a: string, b: string, prompt: string): Promise<-1 | 0 | 1> {
    let res = await this.request(this.sendThink, a, b, prompt);

    if (!res.ok && this.sendThink) {
      const body = await res.text();
      if (/think/i.test(body)) {
        this.sendThink = false;
        res = await this.request(false, a, b, prompt);
      } else {
        throw new Error(`Ollama request failed (${res.status}): ${body}`);
      }
    }

    if (!res.ok) {
      throw new Error(`Ollama request failed (${res.status}): ${await res.text()}`);
    }

    const data = (await res.json()) as { message?: { content?: string } };
    return parseWinner(data.message?.content ?? "");
  }
}
