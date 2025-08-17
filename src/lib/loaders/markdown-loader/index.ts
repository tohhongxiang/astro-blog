import { type Loader, type LoaderContext } from "astro/loaders";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import normalizeRelativePath from "@/lib/normalize-relative-path";
import toTitleCase from "@/lib/to-title-case";
import walk from "@/lib/walk";

import { generateIdFromRelativePath } from "../shared";
import { convertMarkdownToStructured } from "./utils";

export default function markdownLoader(baseDir: string): Loader {
	return {
		name: "markdown-loader",
		async load(context: LoaderContext) {
			const cfgRoot = context.config.root;

			const base = new URL(baseDir, cfgRoot);
			if (!base.pathname.endsWith("/"))
				base.pathname = `${base.pathname}/`;
			const rootDir = fileURLToPath(base);

			async function upsertMarkdown(absPath: string) {
				const relFromBase = normalizeRelativePath(rootDir, absPath);
				const id = generateIdFromRelativePath(relFromBase);

				try {
					const raw = await fs.readFile(absPath, "utf-8");
					const baseName = path.basename(
						absPath,
						path.extname(absPath),
					);
					const result = convertMarkdownToStructured(raw);
					if ("error" in result) {
						context.logger?.warn?.(
							`Skipping invalid markdown: ${relFromBase}`,
						);
						return;
					}

					const content = result.content ?? "";
					const frontmatter = result.frontmatter ?? {};

					if (!frontmatter.title) {
						const firstLine = content.split("\n")[0];
						const firstLineMatch = firstLine?.match(/^#\s*(.*)/);
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

					const rendered = await context.renderMarkdown(content);
					const parsedData = await context.parseData({
						id,
						data: { ...frontmatter, relativeFilePath: relFromBase },
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
						digest: context.generateDigest(content),
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
				const ext = path.extname(abs).toLowerCase();
				if (ext === ".md" || ext === ".mdx") files.push(abs);
			}
			await Promise.all(files.map(upsertMarkdown));

			// Watch mode
			if (!context.watcher) {
				return;
			}

			context.watcher.add(rootDir);

			const onChange = async (changedPath: string) => {
				if (!changedPath.startsWith(rootDir)) return;
				const ext = path.extname(changedPath).toLowerCase();
				if (ext === ".md" || ext === ".mdx")
					await upsertMarkdown(changedPath);
			};

			const onUnlink = async (deletedPath: string) => {
				if (!deletedPath.startsWith(rootDir)) return;
				const ext = path.extname(deletedPath).toLowerCase();
				if (ext === ".md" || ext === ".mdx") {
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
