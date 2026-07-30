#!/usr/bin/env node
import { Command, InvalidArgumentError } from "commander";
import { readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { rankByPrompt } from "./sort.js";
import { loadDotEnv } from "./env.js";
import { createComparator, listOllamaModels, type ComparatorConfig } from "./providers/index.js";
import {
  loadFromFileList,
  loadFromSingleFile,
  unescapeSeparator,
  writeJoinedFile,
  writeSplitFiles,
  type LoadedItem,
} from "./io.js";

interface CliOptions {
  prompt: string;
  input?: string;
  separator: string;
  outputSeparator?: string;
  output?: string;
  outDir?: string;
  keepNames?: boolean;
  extension?: string;
  provider: "ollama" | "openai";
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  think?: boolean;
  concurrency: number;
  refine?: number;
  refineWindow?: number;
  trim: boolean;
  scores?: boolean;
  json?: boolean;
  force?: boolean;
  quiet?: boolean;
}

const DEFAULT_OPENAI_MODEL = "gpt-5.4-mini";

function parsePositiveInt(value: string): number {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n) || n < 1) {
    throw new InvalidArgumentError("must be a positive integer");
  }
  return n;
}

function parseNonNegativeInt(value: string): number {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n) || n < 0) {
    throw new InvalidArgumentError("must be a non-negative integer");
  }
  return n;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/** Refuses to clobber existing output unless --force was passed. */
async function guardOutputs(options: CliOptions): Promise<void> {
  if (options.force) return;

  if (options.output && (await pathExists(resolve(options.output)))) {
    throw new Error(`Output file already exists: ${options.output}\nPass --force to overwrite it.`);
  }

  if (options.outDir) {
    const dir = resolve(options.outDir);
    if (await pathExists(dir)) {
      const entries = await readdir(dir);
      if (entries.length > 0) {
        throw new Error(
          `Output directory is not empty: ${options.outDir} (${entries.length} entries)\n` +
            `Pass --force to write into it anyway.`,
        );
      }
    }
  }
}

async function loadItems(options: CliOptions, files: string[]): Promise<LoadedItem[]> {
  const loadOptions = { separator: unescapeSeparator(options.separator), trim: options.trim };

  if (options.input) {
    if (files.length > 0) {
      throw new Error("Pass either --input <file> or a list of file arguments, not both.");
    }
    return loadFromSingleFile(options.input, loadOptions);
  }

  if (files.length === 0) {
    throw new Error("No input. Pass --input <file> with a separator, or one or more file paths.");
  }

  if (files.length === 1) {
    // A lone file argument is ambiguous; treat it as a single item and say so,
    // since --input is the documented way to split one file into many items.
    process.stderr.write(
      "note: one file argument given, treating it as a single item. " +
        "Use --input to split one file on a separator instead.\n",
    );
  }

  return loadFromFileList(files, loadOptions);
}

async function buildComparatorConfig(options: CliOptions): Promise<ComparatorConfig> {
  if (options.provider === "openai") {
    return {
      provider: "openai",
      model: options.model ?? DEFAULT_OPENAI_MODEL,
      apiKey: options.apiKey,
      baseUrl: options.baseUrl,
    };
  }

  let model = options.model;
  if (!model) {
    // No sensible universal default exists for a local Ollama install, so use
    // whichever model the user pulled most recently.
    const installed = await listOllamaModels(options.baseUrl);
    if (installed.length === 0) {
      throw new Error("No models installed in Ollama. Run `ollama pull <model>` first.");
    }
    model = installed[0];
    if (!options.quiet) {
      process.stderr.write(`using ollama model: ${model}\n`);
    }
  }
  return { provider: "ollama", model, baseUrl: options.baseUrl, think: options.think };
}

async function main(): Promise<void> {
  // Pull OPENAI_API_KEY (and any other keys) from a local .env, if present.
  // Real environment variables take precedence over the file.
  loadDotEnv();

  const program = new Command();

  program
    .name("ai-sort")
    .description(
      "Sort texts by how well each fits a prompt, using an LLM as a pairwise judge.\n" +
        "Best fit comes first.",
    )
    .argument("[files...]", "text files, one item per file")
    .requiredOption("-p, --prompt <prompt>", "the criterion items are ranked against")
    .option("-i, --input <file>", "single file containing all items, split on --separator")
    .option("-s, --separator <sep>", "separator for --input and joined output (escapes allowed)", "\\n")
    .option("--output-separator <sep>", "override --separator when writing joined output")
    .option("-o, --output <file>", "write all items to one file, joined by the separator")
    .option("-d, --out-dir <dir>", "write each item to its own numbered file")
    .option("--keep-names", "in --out-dir, keep source basenames after the rank prefix")
    .option("--extension <ext>", "file extension for --out-dir output", ".txt")
    .option("--provider <name>", "ollama | openai", "ollama")
    .option(
      "-m, --model <model>",
      `model name (default: newest installed Ollama model, or ${DEFAULT_OPENAI_MODEL} for openai)`,
    )
    .option("--base-url <url>", "override provider base URL (e.g. http://localhost:11434)")
    .option("--api-key <key>", "OpenAI API key (defaults to $OPENAI_API_KEY)")
    .option("--think", "let thinking-capable Ollama models reason before judging (slower)")
    .option("-c, --concurrency <n>", "max concurrent LLM comparisons", parsePositiveInt, 4)
    .option(
      "-r, --refine <n>",
      "extra comparisons spent resolving uncertain adjacent pairs (default: item count, 0 to disable)",
      parseNonNegativeInt,
    )
    .option(
      "--refine-window <n>",
      "cap how many ranks apart refined pairs may be (default: unbounded)",
      parsePositiveInt,
    )
    .option("--no-trim", "keep leading/trailing whitespace on each item")
    .option("--scores", "print Bradley-Terry scores alongside stdout output")
    .option("--json", "print the ranking as JSON to stdout")
    .option("-f, --force", "overwrite existing output file / non-empty output directory")
    .option("-q, --quiet", "suppress progress output on stderr")
    .showHelpAfterError();

  program.parse();

  const files = program.args as string[];
  const options = program.opts<CliOptions>();

  if (options.output && options.outDir) {
    throw new Error("Pass either --output <file> or --out-dir <dir>, not both.");
  }

  await guardOutputs(options);

  const items = await loadItems(options, files);
  if (items.length === 0) {
    throw new Error("Input contained no non-empty items.");
  }

  const comparator = createComparator(await buildComparatorConfig(options));
  const texts = items.map((i) => i.text);

  const startedAt = process.hrtime.bigint();
  const result = await rankByPrompt(texts, {
    prompt: options.prompt,
    comparator,
    concurrency: options.concurrency,
    refinementBudget: options.refine,
    refinementWindow: options.refineWindow,
    onProgress: options.quiet
      ? undefined
      : ({ comparisons, phase }) => {
          process.stderr.write(`\r${phase}: ${comparisons} comparisons`.padEnd(48));
        },
  });
  const elapsedMs = Number(process.hrtime.bigint() - startedAt) / 1e6;

  if (!options.quiet) {
    process.stderr.write(
      `\rsorted ${items.length} items with ${result.comparisons} comparisons ` +
        `in ${(elapsedMs / 1000).toFixed(1)}s\n`,
    );
  }

  const ranked = result.order.map((id) => items[id]);
  const rankedScores = result.order.map((id) => result.scores[id]);

  if (options.outDir) {
    const written = await writeSplitFiles(ranked, {
      outDir: options.outDir,
      extension: options.extension,
      keepNames: options.keepNames,
    });
    if (!options.quiet) {
      process.stderr.write(`wrote ${written.length} files to ${resolve(options.outDir)}\n`);
    }
    return;
  }

  if (options.output) {
    const sep = unescapeSeparator(options.outputSeparator ?? options.separator);
    const target = await writeJoinedFile(ranked, options.output, sep);
    if (!options.quiet) {
      process.stderr.write(`wrote ${target}\n`);
    }
    return;
  }

  if (options.json) {
    const payload = ranked.map((item, index) => ({
      rank: index + 1,
      score: rankedScores[index],
      text: item.text,
      ...(item.sourcePath ? { sourcePath: item.sourcePath } : {}),
    }));
    process.stdout.write(`${JSON.stringify({ comparisons: result.comparisons, items: payload }, null, 2)}\n`);
    return;
  }

  const sep = unescapeSeparator(options.outputSeparator ?? options.separator);
  const lines = ranked.map((item, index) =>
    options.scores ? `[${rankedScores[index].toFixed(4)}] ${item.text}` : item.text,
  );
  process.stdout.write(lines.join(sep) + "\n");
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`\nai-sort: ${message}\n`);
  process.exitCode = 1;
});
