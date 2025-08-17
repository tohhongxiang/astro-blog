import type { MarkdownFrontmatter, MarkdownToStructuredResult } from "./types";

export function convertMarkdownToStructured(
	rawContent: string,
): MarkdownToStructuredResult {
	try {
		// Normalize line endings to handle both LF and CRLF
		const normalizedContent = rawContent.replace(/\r\n/g, "\n");

		const frontmatterMatch = normalizedContent.match(
			/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/,
		);

		if (!frontmatterMatch) {
			// No frontmatter found, treat entire content as markdown
			return {
				content: normalizedContent.trim(),
				frontmatter: {},
			};
		}

		const [, frontmatterText, content] = frontmatterMatch;
		const frontmatter = parseFrontmatter(frontmatterText || "");

		return {
			content: (content || "").trim(),
			frontmatter,
		};
	} catch (e) {
		return { error: e };
	}
}

function parseFrontmatter(frontmatterText: string): MarkdownFrontmatter {
	const frontmatter: MarkdownFrontmatter = {};

	// Simple YAML-like parsing for basic key-value pairs
	const lines = frontmatterText.split("\n");

	for (const line of lines) {
		const trimmedLine = line.trim();
		if (!trimmedLine || trimmedLine.startsWith("#")) {
			continue; // Skip empty lines and comments
		}

		const colonIndex = trimmedLine.indexOf(":");
		if (colonIndex === -1) {
			continue; // Skip lines without colons
		}

		const key = trimmedLine.slice(0, colonIndex).trim();
		let value = trimmedLine.slice(colonIndex + 1).trim();

		// Remove quotes if present
		if (
			(value.startsWith('"') && value.endsWith('"')) ||
			(value.startsWith("'") && value.endsWith("'"))
		) {
			value = value.slice(1, -1);
		}

		// Try to parse as date if it looks like a date
		if (key === "date" && /^\d{4}-\d{2}-\d{2}/.test(value)) {
			frontmatter[key] = value;
		} else if (value === "true") {
			frontmatter[key] = true;
		} else if (value === "false") {
			frontmatter[key] = false;
		} else if (/^\d+$/.test(value)) {
			frontmatter[key] = parseInt(value, 10);
		} else if (/^\d+\.\d+$/.test(value)) {
			frontmatter[key] = parseFloat(value);
		} else {
			frontmatter[key] = value;
		}
	}

	return frontmatter;
}
