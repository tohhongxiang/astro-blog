import { promises as fs } from "node:fs";
import path from "node:path";

export type NotebookFrontmatter = {
	title?: string;
	description?: string;
	slug?: string;
	date?: string;
};

export type NotebookToMarkdownOk = {
	markdown: string;
	frontmatter: NotebookFrontmatter;
};

export type NotebookToMarkdownErr = { error: unknown };
export type NotebookToMarkdownResult =
	| NotebookToMarkdownOk
	| NotebookToMarkdownErr;

enum NotebookOutputType {
	Stream = "stream",
	DisplayData = "display_data",
	ExecuteResult = "execute_result",
}

type NotebookStreamOutput = {
	output_type: NotebookOutputType.Stream;
	text?: string | string[];
};

type NotebookDisplayOutput = {
	output_type:
		| NotebookOutputType.DisplayData
		| NotebookOutputType.ExecuteResult;
	data?: Record<string, string | string[] | undefined>;
};

enum CellType {
	Markdown = "markdown",
	Code = "code",
}

type NotebookMarkdownCell = {
	cell_type: CellType.Markdown;
	source?: string | string[];
};

type NotebookCodeCell = {
	cell_type: CellType.Code;
	source?: string | string[];
	metadata?: { language?: string };
	outputs?: Array<NotebookStreamOutput | NotebookDisplayOutput>;
};

type RawNotebook = {
	metadata?: {
		language_info?: { name?: string };
		astro?: NotebookFrontmatter;
	} & NotebookFrontmatter;
	cells?: Array<NotebookMarkdownCell | NotebookCodeCell>;
};

export function toTitleCase(input: string): string {
	return input
		.split(/\s+/)
		.filter(Boolean)
		.map((w) => w.charAt(0).toUpperCase() + w.slice(1))
		.join(" ");
}

export function generateIdFromRelativePath(rel: string): string {
	const noExt = rel.replace(/\.[^/.]+$/, "");
	return noExt.replace(/\\/g, "/");
}

export async function* walk(dir: string): AsyncGenerator<string> {
	const entries = await fs.readdir(dir, { withFileTypes: true });
	for (const entry of entries) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) {
			if (entry.name === ".ipynb_checkpoints") continue;
			yield* walk(full);
		} else if (entry.isFile()) {
			yield full;
		}
	}
}

export function convertNotebookToMarkdown(
	json: string,
	fileNameWithoutExtensions: string,
): NotebookToMarkdownResult {
	let parsed: unknown;
	try {
		parsed = JSON.parse(json);
	} catch (e) {
		return { error: e };
	}

	const nb = parsed as RawNotebook;
	const defaultLang = nb?.metadata?.language_info?.name || "python";
	const frontmatter: NotebookFrontmatter = extractFrontmatter(
		nb,
		fileNameWithoutExtensions,
	);

	const cells = nb?.cells || [];
	const lines: string[] = [];
	for (const cell of cells) {
		if (
			cell.cell_type !== CellType.Markdown &&
			cell.cell_type !== CellType.Code
		) {
			continue; // Skip unsupported cell types
		}

		const src = joinText(cell.source);
		if (cell.cell_type === CellType.Markdown) {
			lines.push(src.trim(), "");
			continue;
		}

		// cell.cell_type === "code"
		const lang = cell?.metadata?.language || defaultLang;

		const outputs = Array.isArray(cell.outputs) ? cell.outputs : [];
		const textOutputs: string[] = [];
		const imageOutputs: string[] = [];

		for (const out of outputs) {
			if (out.output_type === NotebookOutputType.Stream) {
				const processedText = joinText(out.text).replace(/\n$/, "");
				textOutputs.push(processedText);
			} else if (
				out.output_type === NotebookOutputType.DisplayData ||
				out.output_type === NotebookOutputType.ExecuteResult
			) {
				if (out.data?.["image/png"]) {
					const generatedImageMarkdown =
						generateImageMarkdownFromBase64(
							out.data?.["image/png"],
						);
					imageOutputs.push(generatedImageMarkdown);
				}

				if (out.data?.["text/plain"]) {
					const processedText = joinText(
						out.data?.["text/plain"],
					).replace(/\n$/, "");
					textOutputs.push(processedText);
				}
			}
		}

		if (textOutputs.length > 0) {
			lines.push("```" + lang + " withOutput");
			const trimmedSrc = src.replace(/\n$/, "");
			if (trimmedSrc.length > 0) {
				const codeLines = trimmedSrc.split(/\r?\n/);
				for (const codeLine of codeLines) {
					lines.push("> " + codeLine);
				}
			}
			lines.push("");
			lines.push(textOutputs.join("\n"));
			lines.push("```");
		} else {
			lines.push("```" + lang, src.replace(/\n$/, ""), "```");
		}

		for (const image of imageOutputs) {
			lines.push(image);
		}

		lines.push("");
	}

	return { markdown: lines.join("\n"), frontmatter };
}

function extractFrontmatter(
	nb: RawNotebook,
	fileName: string,
): NotebookFrontmatter {
	const fm: NotebookFrontmatter = {};
	const meta = nb.metadata || {};
	const astro = meta.astro || {};

	fm.title = astro.title || meta.title;
	fm.description = astro.description || meta.description;
	fm.slug = astro.slug || meta.slug;
	fm.date = astro.date || meta.date;

	if (!fm.title) {
		const firstCellTitle = getTitleFromCell(nb.cells?.[0]);
		fm.title =
			firstCellTitle || toTitleCase(fileName.replace(/[-_]+/g, " "));
	}
	return fm;
}

function getTitleFromCell(cell?: NotebookMarkdownCell | NotebookCodeCell) {
	if (!cell || cell.cell_type !== CellType.Markdown) {
		return undefined;
	}

	const src = Array.isArray(cell.source)
		? cell.source.join("")
		: String(cell.source ?? "");

	const h1Match = src.match(/^#\s+(.+)$/m);
	if (h1Match && h1Match[1]) return h1Match[1].trim();

	return undefined;
}

function joinText(txt: string | string[] | undefined) {
	return Array.isArray(txt) ? txt.join("") : String(txt ?? "");
}

function generateImageMarkdownFromBase64(
	b64: string | string[] | undefined,
): string {
	if (!b64) return "";
	const base64String = joinText(b64);
	return `![output](data:image/png;base64,${base64String})`;
}
