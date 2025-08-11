// @ts-check
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";
import expressiveCode from "astro-expressive-code";
import { defineConfig } from "astro/config";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeExternalLinks from "rehype-external-links";
import rehypeKatex from "rehype-katex";
import rehypeMermaid from "rehype-mermaid";
import rehypeSlug from "rehype-slug";
import remarkMath from "remark-math";
import { visit } from "unist-util-visit";

import { pluginCodeOutput } from "./src/lib/plugins/expressive-code-output";

// https://astro.build/config
export default defineConfig({
	integrations: [
		react(),
		expressiveCode({
			themes: ["one-dark-pro", "one-light"],
			themeCssSelector: (theme) => `[data-code-theme='${theme.name}']`,
			plugins: [pluginCodeOutput()],
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
			[
				rehypeMermaid,
				{
					strategy: "img-svg",
					dark: true,
				},
			],
			rehypeModifyMermaidGraphs,
			rehypeSlug,
			[
				rehypeAutolinkHeadings,
				{ behavior: "prepend", properties: { class: "header-link" } },
			],
		],
	},
});

function rehypeModifyMermaidGraphs() {
	return (tree) => {
		visit(tree, "element", (node) => {
			if (node.tagName === "picture") {
				// if has children <source> and <img> with a tag prefix of "mermaid-[index]"
				if (
					node.children.length === 2 &&
					node.children[0].tagName === "source" &&
					node.children[1].tagName === "img"
				) {
					const sourceNode = node.children[0];
					const imgNode = node.children[1];

					// check the prefix
					const split = imgNode.properties.id.split("-");
					if (split.length !== 2) return;
					if (split[0] !== "mermaid") return;

					const mermaidIndex = parseInt(split[1]);
					const darkID = `mermaid-dark-${mermaidIndex}`;
					const ID = `mermaid-${mermaidIndex}`;

					// skip if the children IDs don't match
					if (
						sourceNode.properties.id !== darkID ||
						imgNode.properties.id !== ID
					) {
						return;
					}

					// add mx-auto to both source and img
					sourceNode.properties.className =
						sourceNode.properties.className || [];
					sourceNode.properties.className.push("mx-auto");
					sourceNode.properties.className.push("mermaid-dark");

					imgNode.properties.className =
						imgNode.properties.className || [];
					imgNode.properties.className.push("mx-auto");
					imgNode.properties.className.push("mermaid");
				}
			}
		});
	};
}
