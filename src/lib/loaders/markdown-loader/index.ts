import {
	type Loader,
	type LoaderContext,
	glob as globLoader,
} from "astro/loaders";
import { fileURLToPath } from "node:url";

import normalizeRelativePath from "@/lib/normalize-relative-path";

export default function markdownLoader(baseDir: string): Loader {
	return {
		name: "markdown-loader",
		async load(context: LoaderContext) {
			const base = new URL(baseDir, context.config.root);

			// Use Astro's proven glob loader for the core functionality
			const mdGlob = globLoader({
				pattern: ["**/*.md", "**/*.mdx"],
				base,
			});
			await mdGlob.load(context);

			// Add our custom metadata to each entry
			const allEntries = Array.from(context.store.entries());
			const baseDirPath = fileURLToPath(base);

			allEntries.forEach(([, entry]) => {
				if (entry.filePath) {
					const relFromBase = normalizeRelativePath(
						baseDirPath,
						entry.filePath,
					);

					// Add relativeFilePath to the data
					entry.data = {
						...entry.data,
						relativeFilePath: relFromBase,
					};

					// Update the store with the modified entry
					context.store.set(entry);
				}
			});
		},
	};
}
