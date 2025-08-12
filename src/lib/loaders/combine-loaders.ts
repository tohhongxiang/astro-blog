import type { Loader } from "astro/loaders";

export default function combineLoaders(...loaders: Loader[]): Loader {
	return {
		name: loaders.map((l) => l.name).join(" + "),
		async load(context) {
			await Promise.all(loaders.map((loader) => loader.load(context)));
		},
	};
}
