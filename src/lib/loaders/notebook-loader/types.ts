export type NotebookFrontmatter = {
	title?: string;
	description?: string;
	slug?: string;
	date?: string;
};

export type NotebookToMarkdownOk = {
	markdown: string;
	frontmatter: NotebookFrontmatter;
};

export type NotebookToMarkdownErr = { error: unknown };
export type NotebookToMarkdownResult =
	| NotebookToMarkdownOk
	| NotebookToMarkdownErr;

export enum NotebookOutputType {
	Stream = "stream",
	DisplayData = "display_data",
	ExecuteResult = "execute_result",
}

export type NotebookStreamOutput = {
	output_type: NotebookOutputType.Stream;
	text?: string | string[];
};

export type NotebookDisplayOutput = {
	output_type:
		| NotebookOutputType.DisplayData
		| NotebookOutputType.ExecuteResult;
	data?: Record<string, string | string[] | undefined>;
};

export enum CellType {
	Markdown = "markdown",
	Code = "code",
}

export type NotebookMarkdownCell = {
	cell_type: CellType.Markdown;
	source?: string | string[];
};

export type NotebookCodeCell = {
	cell_type: CellType.Code;
	source?: string | string[];
	metadata?: { language?: string };
	outputs?: Array<NotebookStreamOutput | NotebookDisplayOutput>;
};

export type RawNotebook = {
	metadata?: {
		language_info?: { name?: string };
		astro?: NotebookFrontmatter;
	} & NotebookFrontmatter;
	cells?: Array<NotebookMarkdownCell | NotebookCodeCell>;
};
