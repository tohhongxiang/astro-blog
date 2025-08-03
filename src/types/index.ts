import type { CollectionEntry } from "astro:content";

export type Post = CollectionEntry<"blog">;
export type Note = CollectionEntry<"notes">;
