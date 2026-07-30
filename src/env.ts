import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Loads a `.env` file into `process.env` without adding a dependency.
 *
 * Real environment variables always win: a key already present in
 * `process.env` is never overwritten. Missing files are ignored silently, so
 * this is safe to call unconditionally at startup. Only used by the CLI — the
 * library never touches the filesystem or the environment on import.
 *
 * Supports `KEY=value`, `export KEY=value`, `#` comments, blank lines, and
 * single- or double-quoted values (with `\n`/`\t` escapes inside double quotes).
 */
export function loadDotEnv(path = ".env"): void {
  let contents: string;
  try {
    contents = readFileSync(resolve(path), "utf8");
  } catch {
    return; // no .env file, nothing to do
  }

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const withoutExport = line.startsWith("export ") ? line.slice(7).trim() : line;
    const eq = withoutExport.indexOf("=");
    if (eq === -1) continue;

    const key = withoutExport.slice(0, eq).trim();
    if (!key || key in process.env) continue;

    process.env[key] = parseValue(withoutExport.slice(eq + 1).trim());
  }
}

function parseValue(raw: string): string {
  if (raw.length >= 2 && raw[0] === '"' && raw.endsWith('"')) {
    return raw.slice(1, -1).replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\"/g, '"');
  }
  if (raw.length >= 2 && raw[0] === "'" && raw.endsWith("'")) {
    return raw.slice(1, -1);
  }
  // Unquoted: strip a trailing inline comment (preceded by whitespace).
  return raw.replace(/\s+#.*$/, "").trim();
}
