import { type Loader } from "astro/loaders";
import { promises as fs } from "node:fs";
import path from "node:path";

import normalizeRelativePath from "@/lib/normalize-relative-path";
import toTitleCase from "@/lib/to-title-case";

import globWithParser from "../glob-with-parser";

export default function markdownLoader(baseDir: string): Loader {
	return globWithParser({
		pattern: ["**/*.md", "**/*.mdx"],
		base: baseDir,
		parser: async (entry) => {
			if (!entry.filePath) {
				return entry;
			}

			const relToBaseDir = normalizeRelativePath(baseDir, entry.filePath);
			(entry.data as Record<string, unknown>).relativeFilePath =
				relToBaseDir;

			if (!entry.data.title) {
				const raw = await fs.readFile(entry.filePath, "utf-8");
				const firstLine = raw
					.replace(/^---\r?\n[\s\S\r\n]*?---(\r?\n)*/, "")
					.split("\n")[0];
				const firstLineMatch = firstLine?.match(/^#\s*(.*)/);

				(entry.data as Record<string, unknown>).title =
					firstLineMatch?.[1]?.trim() ||
					toTitleCase(
						entry.id.split(path.sep).pop()?.replace(/-/g, " ") ??
							"",
					);
			}

			if (!entry.data.date) {
				let date = new Date().toISOString();
				try {
					const st = await fs.stat(entry.filePath);
					date = new Date(st.mtime).toISOString();
				} catch {
					date = new Date().toISOString();
				}

				(entry.data as Record<string, unknown>).date = date;
			}

			return entry;
		},
	});
}
