import { ChevronRight } from "lucide-react";

import { formatDate } from "@/lib/format-date";
import type { Note } from "@/types";

export default function NoteItem({ note }: { note: Note }) {
	return (
		<a
			href={`/${note.collection}/${note.id}`}
			aria-label={note.data.title}
			className="group block rounded-md px-4 py-3 transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
		>
			<div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-4">
				{note.data.date && (
					<time
						dateTime={formatDate(note.data.date)}
						className="hidden shrink-0 tabular-nums font-mono text-muted-foreground sm:block"
					>
						{formatDate(note.data.date)}
					</time>
				)}
				<div className="flex flex-1 flex-col gap-1">
					<span className="font-semibold group-hover:underline">
						{note.data.title}
					</span>
					{note.data.relativeFilePath && (
						<p className="text-xs text-muted-foreground">
							{note.data.relativeFilePath
								.split("/")
								.filter(Boolean)
								.join(" > ")}
						</p>
					)}
				</div>
				{/* Chevron indicator on desktop */}
				<ChevronRight
					className="ml-auto hidden h-4 w-4 self-center text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 sm:block"
					aria-hidden="true"
				/>
				{note.data.date && (
					<time
						dateTime={formatDate(note.data.date)}
						className="tabular-nums font-mono text-xs text-muted-foreground sm:hidden"
					>
						{formatDate(note.data.date)}
					</time>
				)}
			</div>
		</a>
	);
}
