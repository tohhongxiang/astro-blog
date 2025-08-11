import {
	type Loader,
	type LoaderContext,
	glob as globLoader,
} from "astro/loaders";
import { defineCollection, z } from "astro:content";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
	type NotebookFrontmatter,
	type NotebookToMarkdownOk,
	convertNotebookToMarkdown,
	generateIdFromRelativePath,
	walk,
} from "./lib/content/notebook-utils";

function notebooksAndMarkdownLoader(baseDir: string): Loader {
	return {
		name: "md-ipynb-inline",
		async load(context: LoaderContext) {
			const cfgRoot = context.config.root;

			const base = new URL(baseDir, cfgRoot);
			if (!base.pathname.endsWith("/"))
				base.pathname = `${base.pathname}/`;
			const rootDir = fileURLToPath(base);

			// Delegate .md/.mdx to glob loader
			const mdGlob = globLoader({
				pattern: ["**/*.md", "**/*.mdx"],
				base,
			});
			await mdGlob.load(context);

			async function upsertIpynb(absPath: string) {
				const relFromBase = path
					.relative(rootDir, absPath)
					.split(path.sep)
					.join("/");
				const id = generateIdFromRelativePath(relFromBase);
				const relToRoot = path
					.relative(fileURLToPath(cfgRoot), absPath)
					.split(path.sep)
					.join("/");
				try {
					const raw = await fs.readFile(absPath, "utf-8");
					const baseName = path.basename(absPath, ".ipynb");
					const result = convertNotebookToMarkdown(raw, baseName);
					if ("error" in result) {
						context.logger?.warn?.(
							`Skipping invalid notebook (JSON): ${relFromBase}`,
						);
						return;
					}
					const markdown =
						(result as NotebookToMarkdownOk).markdown ?? "";
					const frontmatter: NotebookFrontmatter =
						(result as NotebookToMarkdownOk).frontmatter ?? {};
					if (!("date" in frontmatter) || !frontmatter.date) {
						try {
							const st = await fs.stat(absPath);
							(frontmatter as { date: string }).date = new Date(
								st.mtime,
							).toISOString();
						} catch {
							(frontmatter as { date: string }).date =
								new Date().toISOString();
						}
					}
					const rendered = await context.renderMarkdown(markdown);
					const parsedData = await context.parseData({
						id,
						data: frontmatter,
						filePath: absPath,
					});
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

			// initial load
			for await (const abs of walk(rootDir)) {
				if (abs.toLowerCase().endsWith(".ipynb"))
					await upsertIpynb(abs);
			}

			// watch for changes in dev
			if (context.watcher) {
				context.watcher.add(rootDir);
				const onChange = async (changedPath: string) => {
					if (!changedPath.startsWith(rootDir)) return;
					if (changedPath.toLowerCase().endsWith(".ipynb"))
						await upsertIpynb(changedPath);
				};
				const onUnlink = async (deletedPath: string) => {
					if (!deletedPath.startsWith(rootDir)) return;
					if (deletedPath.toLowerCase().endsWith(".ipynb")) {
						const relFromBase = path
							.relative(rootDir, deletedPath)
							.split(path.sep)
							.join("/");
						const id = generateIdFromRelativePath(relFromBase);
						context.store.delete(id);
					}
				};
				context.watcher.on("change", onChange);
				context.watcher.on("add", onChange);
				context.watcher.on("unlink", onUnlink);
			}
		},
	};
}

const blog = defineCollection({
	loader: notebooksAndMarkdownLoader("./src/collections/blog"),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
	}),
});

const notes = defineCollection({
	loader: notebooksAndMarkdownLoader("./src/collections/notes"),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
	}),
});

export const collections = { blog, notes };
