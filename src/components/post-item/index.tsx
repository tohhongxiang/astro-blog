import { ChevronRight } from "lucide-react";

import { formatDate } from "@/lib/format-date";
import type { Post } from "@/types";

export default function PostItem({ post }: { post: Post }) {
	return (
		<a
			href={`/${post.collection}/${post.id}`}
			aria-label={post.data.title}
			className="group block rounded-md px-4 py-3 transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
		>
			<div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-4">
				{post.data.date && (
					<time
						dateTime={formatDate(post.data.date)}
						className="hidden shrink-0 tabular-nums font-mono text-muted-foreground sm:block"
					>
						{formatDate(post.data.date)}
					</time>
				)}
				<div className="flex flex-1 items-center gap-2 sm:gap-0">
					<span className="font-semibold group-hover:underline">
						{post.data.title}
					</span>
					{/* Chevron indicator on desktop */}
					<ChevronRight
						className="ml-auto hidden h-4 w-4 self-center text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 sm:block"
						aria-hidden="true"
					/>
				</div>
				{post.data.date && (
					<time
						dateTime={formatDate(post.data.date)}
						className="tabular-nums font-mono text-xs text-muted-foreground sm:hidden"
					>
						{formatDate(post.data.date)}
					</time>
				)}
			</div>
		</a>
	);
}
