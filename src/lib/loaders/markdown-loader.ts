import {
	type Loader,
	type LoaderContext,
	glob as globLoader,
} from "astro/loaders";

export default function markdownLoader(baseDir: string): Loader {
	return {
		name: "markdown-loader",
		async load(context: LoaderContext) {
			const cfgRoot = context.config.root;
			const base = new URL(baseDir, cfgRoot);

			// Run glob loader with correct base path
			const mdGlob = globLoader({
				pattern: ["**/*.md", "**/*.mdx"],
				base,
			});
			await mdGlob.load(context);
		},
	};
}
