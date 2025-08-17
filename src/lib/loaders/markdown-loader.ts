import {
	type Loader,
	type LoaderContext,
	glob as globLoader,
} from "astro/loaders";

import normalizeRelativePath from "../normalize-relative-path";

export default function markdownLoader(baseDir: string): Loader {
	return {
		name: "markdown-loader",
		async load(context: LoaderContext) {
			const cfgRoot = context.config.root;
			const base = new URL(baseDir, cfgRoot);

			const mdGlob = globLoader({
				pattern: ["**/*.md", "**/*.mdx"],
				base,
			});
			await mdGlob.load(context);

			const allEntries = Array.from(context.store.entries());

			allEntries.forEach(([, entry]) => {
				if (entry.filePath) {
					const relFromBase = normalizeRelativePath(
						baseDir,
						entry.filePath,
					);

					// Add relativePath to the data
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
