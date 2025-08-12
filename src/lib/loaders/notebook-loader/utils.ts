import {
	CellType,
	type NotebookFrontmatter,
	NotebookOutputType,
	type NotebookToMarkdownResult,
	type RawNotebook,
} from "./types";

export function generateIdFromRelativePath(rel: string): string {
	const noExt = rel.replace(/\.[^/.]+$/, "");
	return noExt.replace(/\\/g, "/");
}

export function convertNotebookToMarkdown(
	json: string,
): NotebookToMarkdownResult {
	let parsed: unknown;
	try {
		parsed = JSON.parse(json);
	} catch (e) {
		return { error: e };
	}

	const nb = parsed as RawNotebook;
	const defaultLang = nb?.metadata?.language_info?.name || "python";
	const frontmatter = extractFrontmatter(nb);

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

function extractFrontmatter(nb: RawNotebook): NotebookFrontmatter {
	const fm: NotebookFrontmatter = {};
	const meta = nb.metadata || {};
	const astro = meta.astro || {};

	fm.title = astro.title || meta.title;
	fm.description = astro.description || meta.description;
	fm.slug = astro.slug || meta.slug;
	fm.date = astro.date || meta.date;

	return fm;
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
