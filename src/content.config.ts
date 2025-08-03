import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const blog = defineCollection({
	loader: glob({ pattern: "**/*.md", base: "./src/collections/blog" }),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
	}),
});

const notes = defineCollection({
	loader: glob({ pattern: "**/*.md", base: "./src/collections/notes" }),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
	}),
});

export const collections = { blog, notes };
