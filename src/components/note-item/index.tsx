import { formatDate } from "@/lib/format-date";
import type { Note } from "@/types";

export default function NoteItem({ note }: { note: Note }) {
	return (
		<div>
			<div className="flex flex-row items-baseline gap-4 py-2">
				{note.data.date && (
					<time
						dateTime={formatDate(note.data.date)}
						className="text-muted-foreground font-mono"
					>
						{formatDate(note.data.date)}
					</time>
				)}
				<div className="flex flex-col gap-1">
					<a
						href={`${note.collection}/${note.id}`}
						className="font-semibold hover:underline"
					>
						{note.data.title}
					</a>
					{note.filePath && (
						<p className="text-xs text-muted-foreground">
							{note.filePath
								.split("/")
								.filter(Boolean)
								.slice(1)
								.join(" > ")}
						</p>
					)}
				</div>
			</div>
		</div>
	);
}
