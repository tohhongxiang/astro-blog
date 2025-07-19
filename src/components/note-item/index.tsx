import { formatDate } from "@/lib/format-date";
import type { Post } from "@/types";

export default function NoteItem({ note }: { note: Post }) {
	return (
		<div>
			<div className="flex flex-row items-baseline gap-4 py-2">
				{note.frontmatter.date && (
					<time
						dateTime={formatDate(note.frontmatter.date)}
						className="text-muted-foreground font-mono"
					>
						{formatDate(note.frontmatter.date)}
					</time>
				)}
				<div className="flex flex-col gap-1">
					<a
						href={note.url}
						className="font-semibold hover:underline"
					>
						{note.frontmatter.title}
					</a>
					<p className="text-xs text-muted-foreground">
						{note.url
							.split("/")
							.filter(Boolean)
							.slice(1)
							.join(" > ")}
					</p>
				</div>
			</div>
		</div>
	);
}
