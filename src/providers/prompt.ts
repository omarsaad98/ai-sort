export const COMPARE_SYSTEM_PROMPT =
  "You judge which of two candidate texts better fits a given prompt/criterion. " +
  'Respond with strict JSON only, no markdown, no commentary: {"winner": "A" | "B" | "tie"}. ' +
  '"A" means candidate A fits the prompt better, "B" means candidate B does, "tie" means they fit equally well.';

export function buildCompareUserMessage(prompt: string, a: string, b: string): string {
  return [
    `Prompt/criterion: ${prompt}`,
    "",
    "Candidate A:",
    a,
    "",
    "Candidate B:",
    b,
    "",
    'Which candidate fits the prompt better? Respond with JSON only: {"winner": "A" | "B" | "tie"}',
  ].join("\n");
}

export function parseWinner(raw: string): -1 | 0 | 1 {
  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  const text = jsonMatch ? jsonMatch[0] : raw;
  try {
    const parsed = JSON.parse(text);
    const winner = String(parsed.winner ?? "").trim().toLowerCase();
    if (winner === "a") return 1;
    if (winner === "b") return -1;
    if (winner === "tie") return 0;
  } catch {
    // fall through to loose text matching below
  }
  const loose = raw.trim().toLowerCase();
  if (loose.startsWith("a")) return 1;
  if (loose.startsWith("b")) return -1;
  return 0;
}
