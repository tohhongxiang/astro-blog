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

type NotebookStreamOutput = {
	output_type: "stream";
	text?: string | string[];
};

type NotebookDisplayOutput = {
	output_type: "display_data" | "execute_result";
	data?: Record<string, unknown>;
};

type NotebookMarkdownCell = {
	cell_type: "markdown";
	source?: string | string[];
};

type NotebookCodeCell = {
	cell_type: "code";
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
	fallbackBaseName: string,
): NotebookToMarkdownResult {
	let parsed: unknown;
	try {
		parsed = JSON.parse(json);
	} catch (e) {
		return { error: e };
	}

	const nb = parsed as RawNotebook;
	const defaultLang = nb?.metadata?.language_info?.name || "python";
	const frontmatter: NotebookFrontmatter = {};
	if (nb?.metadata) {
		const meta = nb.metadata;
		const astro = meta.astro || {};
		frontmatter.title = astro.title || meta.title;
		frontmatter.description = astro.description || meta.description;
		frontmatter.slug = astro.slug || meta.slug;
		frontmatter.date = astro.date || meta.date;
	}

	const lines: string[] = [];
	for (const cell of nb?.cells || []) {
		if (cell.cell_type === "markdown") {
			const src = Array.isArray(cell.source)
				? cell.source.join("")
				: String(cell.source ?? "");
			if (!frontmatter.title) {
				const m = src.match(/^#\s+(.+)$/m);
				if (m && m[1]) frontmatter.title = m[1].trim();
			}
			lines.push(src.trim(), "");
		} else if (cell.cell_type === "code") {
			const src = Array.isArray(cell.source)
				? cell.source.join("")
				: String(cell.source ?? "");
			const lang = cell?.metadata?.language || defaultLang;

			const outputs = Array.isArray(cell.outputs) ? cell.outputs : [];
			const textOutputs: string[] = [];
			const imageOutputs: string[] = [];

			for (const out of outputs) {
				if (out.output_type === "stream") {
					const txt = Array.isArray(out.text)
						? out.text.join("")
						: String(out.text ?? "");
					textOutputs.push(txt.replace(/\n$/, ""));
				} else if (
					out.output_type === "display_data" ||
					out.output_type === "execute_result"
				) {
					const dataObj = out.data || {};
					const png = dataObj["image/png"] as
						| string
						| string[]
						| undefined;
					const text = dataObj["text/plain"] as
						| string
						| string[]
						| undefined;
					if (png) {
						const b64 = Array.isArray(png)
							? png.join("")
							: String(png);
						imageOutputs.push(
							`![output](data:image/png;base64,${b64})`,
						);
					}
					if (text) {
						const txt = Array.isArray(text)
							? text.join("")
							: String(text);
						textOutputs.push(txt.replace(/\n$/, ""));
					}
				}
			}

			if (textOutputs.length > 0) {
				lines.push("```" + lang + " withOutput");
				const trimmedSrc = src.replace(/\n$/, "");
				if (trimmedSrc.length > 0) {
					const codeLines = trimmedSrc.split(/\r?\n/);
					for (const codeLine of codeLines) {
						if (codeLine.length > 0) {
							lines.push("> " + codeLine);
						} else {
							lines.push(">");
						}
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
	}

	if (!frontmatter.title && fallbackBaseName) {
		frontmatter.title = toTitleCase(
			fallbackBaseName.replace(/[-_]+/g, " "),
		);
	}

	return { markdown: lines.join("\n"), frontmatter };
}
