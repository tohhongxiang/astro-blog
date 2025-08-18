import { Search, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import NoteItem from "@/components/note-item";
import type { Note } from "@/types";

interface SearchProps {
	notes: Note[];
}

export default function NotesSearch({ notes }: SearchProps) {
	const [query, setQuery] = useState("");

	// Read initial query from URL on mount
	useEffect(() => {
		if (typeof window !== "undefined") {
			const urlParams = new URLSearchParams(window.location.search);
			setQuery(urlParams.get("q") || "");
		}
	}, []);

	const searchResults = useMemo(() => {
		if (!query.trim()) return notes;

		const searchTerm = query.toLowerCase();
		return notes.filter(
			(note) =>
				(note.data.title?.toLowerCase() || "").includes(searchTerm) ||
				note.id.toLowerCase().includes(searchTerm),
		);
	}, [query, notes]);

	const updateSearchQuery = (newQuery: string) => {
		setQuery(newQuery);

		if (typeof window !== "undefined") {
			const url = new URL(window.location.href);
			if (newQuery.trim()) {
				url.searchParams.set("q", newQuery);
			} else {
				url.searchParams.delete("q");
			}
			window.history.replaceState({}, "", url.toString());
		}
	};

	return (
		<div className="space-y-2">
			{/* Search Input */}
			<div className="relative">
				<Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
				<input
					type="text"
					placeholder="Search notes by title or path..."
					value={query}
					onChange={(e) => updateSearchQuery(e.target.value)}
					className="w-full rounded-md border border-input bg-background px-10 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/50"
				/>
				{query && (
					<button
						onClick={() => updateSearchQuery("")}
						className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
						aria-label="Clear search"
					>
						<X className="h-4 w-4" />
					</button>
				)}
			</div>

			{/* Search Count */}
			{query && (
				<div className="text-xs text-muted-foreground">
					{searchResults.length} of {notes.length} note(s)
				</div>
			)}

			{/* Results */}
			{query && searchResults.length === 0 ? (
				<div className="text-center py-8 px-4">
					<div className="text-foreground text-md">
						No notes found matching "{query}"
					</div>
					<div className="text-muted-foreground text-xs mt-1">
						Try adjusting your search terms or check the spelling
					</div>
				</div>
			) : (
				<div className="space-y-1 -mx-4">
					{searchResults.map((note) => (
						<NoteItem key={note.id} note={note} />
					))}
				</div>
			)}
		</div>
	);
}
