export type MarkdownFrontmatter = {
	title?: string;
	description?: string;
	slug?: string;
	date?: string;
	[key: string]: unknown;
};

export type MarkdownToStructuredOk = {
	content: string;
	frontmatter: MarkdownFrontmatter;
};

export type MarkdownToStructuredErr = { error: unknown };
export type MarkdownToStructuredResult =
	| MarkdownToStructuredOk
	| MarkdownToStructuredErr;
