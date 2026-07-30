import { readFile, writeFile, mkdir } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";

/**
 * Interprets common backslash escapes in a user-supplied separator, so
 * `--separator "\n---\n"` works from any shell without literal newlines.
 */
export function unescapeSeparator(raw: string): string {
  return raw.replace(/\\(n|r|t|0|\\)/g, (_, ch: string) => {
    switch (ch) {
      case "n":
        return "\n";
      case "r":
        return "\r";
      case "t":
        return "\t";
      case "0":
        return "\0";
      default:
        return "\\";
    }
  });
}

export interface LoadedItem {
  text: string;
  /** Absolute path this item came from, when input was one file per item. */
  sourcePath?: string;
}

export interface LoadOptions {
  separator: string;
  trim: boolean;
}

/** Reads one file and splits it into items on `separator`. */
export async function loadFromSingleFile(path: string, options: LoadOptions): Promise<LoadedItem[]> {
  const content = await readFile(resolve(path), "utf8");
  const parts = content.split(options.separator);
  return parts
    .map((text) => (options.trim ? text.trim() : text))
    .filter((text) => text.length > 0)
    .map((text) => ({ text }));
}

/** Reads each path as one whole item. */
export async function loadFromFileList(paths: string[], options: LoadOptions): Promise<LoadedItem[]> {
  const items = await Promise.all(
    paths.map(async (p) => {
      const abs = resolve(p);
      const raw = await readFile(abs, "utf8");
      return { text: options.trim ? raw.trim() : raw, sourcePath: abs };
    }),
  );
  return items.filter((item) => item.text.length > 0);
}

export interface WriteSplitOptions {
  outDir: string;
  /** Extension for generated files, including the dot. Default ".txt". */
  extension?: string;
  /** Keep the original file's basename after the numeric prefix, when known. */
  keepNames?: boolean;
}

/**
 * Writes each ranked item to its own file, prefixed with a zero-padded rank so
 * a lexical directory listing matches the ranking.
 */
export async function writeSplitFiles(
  items: LoadedItem[],
  options: WriteSplitOptions,
): Promise<string[]> {
  const dir = resolve(options.outDir);
  await mkdir(dir, { recursive: true });
  const width = String(items.length).length;
  const written: string[] = [];

  await Promise.all(
    items.map(async (item, index) => {
      const rank = String(index + 1).padStart(width, "0");
      let name: string;
      if (options.keepNames && item.sourcePath) {
        const src = basename(item.sourcePath);
        const ext = extname(src);
        name = `${rank}-${basename(src, ext)}${ext || options.extension || ".txt"}`;
      } else {
        name = `${rank}${options.extension ?? ".txt"}`;
      }
      const target = join(dir, name);
      await writeFile(target, item.text, "utf8");
      written[index] = target;
    }),
  );

  return written;
}

/** Writes all ranked items into one file, joined by `separator`. */
export async function writeJoinedFile(
  items: LoadedItem[],
  outPath: string,
  separator: string,
): Promise<string> {
  const target = resolve(outPath);
  await writeFile(target, items.map((i) => i.text).join(separator), "utf8");
  return target;
}
