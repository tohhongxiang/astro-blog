import { type Loader, type LoaderContext } from "astro/loaders";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import normalizeRelativePath from "@/lib/normalize-relative-path";
import toTitleCase from "@/lib/to-title-case";
import walk from "@/lib/walk";

import { convertNotebookToMarkdown, generateIdFromRelativePath } from "./utils";

export default function notebookLoader(baseDir: string): Loader {
	return {
		name: "ipynb-loader",
		async load(context: LoaderContext) {
			const cfgRoot = context.config.root;

			const base = new URL(baseDir, cfgRoot);
			if (!base.pathname.endsWith("/"))
				base.pathname = `${base.pathname}/`;
			const rootDir = fileURLToPath(base);

			async function upsertIpynb(absPath: string) {
				const relFromBase = normalizeRelativePath(rootDir, absPath);
				const id = generateIdFromRelativePath(relFromBase);

				try {
					const raw = await fs.readFile(absPath, "utf-8");
					const baseName = path.basename(absPath, ".ipynb");
					const result = convertNotebookToMarkdown(raw);
					if ("error" in result) {
						context.logger?.warn?.(
							`Skipping invalid notebook (JSON): ${relFromBase}`,
						);
						return;
					}

					const markdown = result.markdown ?? "";
					const frontmatter = result.frontmatter ?? {};

					if (!frontmatter.title) {
						const firstLine = markdown.split("\n")[0];
						const firstLineMatch = firstLine.match(/^#\s*(.*)/);
						frontmatter.title =
							firstLineMatch?.[1] ||
							toTitleCase(baseName.replace(/[-_]+/g, " "));
					}

					if (!frontmatter.date) {
						try {
							const st = await fs.stat(absPath);
							frontmatter.date = new Date(st.mtime).toISOString();
						} catch {
							frontmatter.date = new Date().toISOString();
						}
					}

					const rendered = await context.renderMarkdown(markdown);
					const parsedData = await context.parseData({
						id,
						data: frontmatter,
						filePath: absPath,
					});

					const relToRoot = normalizeRelativePath(
						fileURLToPath(cfgRoot),
						absPath,
					);
					context.store.set({
						id,
						data: parsedData,
						body: rendered.html,
						filePath: relToRoot,
						digest: context.generateDigest(markdown),
						rendered,
						assetImports: rendered.metadata?.imagePaths,
					});
				} catch (e) {
					const message = e instanceof Error ? e.message : String(e);
					context.logger?.error?.(
						`Error processing ${relFromBase}: ${message}`,
					);
				}
			}

			// Initial load
			const files: string[] = [];
			for await (const abs of walk(rootDir)) {
				if (abs.startsWith(".")) continue;
				if (abs.toLowerCase().endsWith(".ipynb")) files.push(abs);
			}
			await Promise.all(files.map(upsertIpynb));

			// Watch mode
			if (!context.watcher) {
				return;
			}

			context.watcher.add(rootDir);

			const onChange = async (changedPath: string) => {
				if (!changedPath.startsWith(rootDir)) return;
				if (changedPath.toLowerCase().endsWith(".ipynb"))
					await upsertIpynb(changedPath);
			};

			const onUnlink = async (deletedPath: string) => {
				if (!deletedPath.startsWith(rootDir)) return;
				if (deletedPath.toLowerCase().endsWith(".ipynb")) {
					const relFromBase = normalizeRelativePath(
						rootDir,
						deletedPath,
					);
					const id = generateIdFromRelativePath(relFromBase);
					context.store.delete(id);
				}
			};

			context.watcher.on("change", onChange);
			context.watcher.on("add", onChange);
			context.watcher.on("unlink", onUnlink);
		},
	};
}
