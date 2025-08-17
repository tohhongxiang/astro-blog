import { defineCollection, z } from "astro:content";

import combineLoaders from "./lib/loaders/combine-loaders";
import markdownLoader from "./lib/loaders/markdown-loader";
import notebookLoader from "./lib/loaders/notebook-loader";

// Collections
const blog = defineCollection({
	loader: combineLoaders(
		markdownLoader("./src/collections/blog"),
		notebookLoader("./src/collections/blog"),
	),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
		relativeFilePath: z.string().optional(),
	}),
});

const notes = defineCollection({
	loader: combineLoaders(
		markdownLoader("./src/collections/notes"),
		notebookLoader("./src/collections/notes"),
	),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
		relativeFilePath: z.string().optional(),
	}),
});

export const collections = { blog, notes };
