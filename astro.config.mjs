// @ts-check
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";
import expressiveCode from "astro-expressive-code";
import astroMermaid from "astro-mermaid";
import { defineConfig } from "astro/config";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeExternalLinks from "rehype-external-links";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import remarkMath from "remark-math";

import { pluginCodeOutput } from "./src/lib/plugins/expressive-code-output";

// https://astro.build/config
export default defineConfig({
	integrations: [
		react(),
		astroMermaid({
			theme: "forest",
			autoTheme: true,
		}),
		expressiveCode({
			themes: ["one-dark-pro", "one-light"],
			themeCssSelector: (theme) => `[data-theme='${theme.type}']`,
			useDarkModeMediaQuery: false,
			styleOverrides: {
				borderRadius: "0.625rem",
				codeFontFamily: "var(--font-mono, ui-monospace, monospace)",
			},
			plugins: [pluginCodeOutput()],
			emitExternalStylesheet: false, // Prevent theme flash when loading external stylesheet
		}),
	],
	vite: {
		plugins: [tailwindcss()],
	},
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [
			[
				rehypeExternalLinks,
				{
					rel: ["nofollow", "noopener", "noreferrer"],
					target: ["_blank"],
				},
			],
			[rehypeKatex, {}],
			rehypeSlug,
			[
				rehypeAutolinkHeadings,
				{ behavior: "prepend", properties: { class: "header-link" } },
			],
		],
	},
});
