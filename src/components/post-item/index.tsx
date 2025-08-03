import { formatDate } from "@/lib/format-date";
import type { Post } from "@/types";

export default function PostItem({ post }: { post: Post }) {
	return (
		<div className="flex flex-row items-baseline gap-4 py-2">
			{post.data.date && (
				<time
					dateTime={formatDate(post.data.date)}
					className="text-muted-foreground font-mono"
				>
					{formatDate(post.data.date)}
				</time>
			)}
			<a
				href={`/${post.collection}/${post.id}`}
				className="font-semibold hover:underline"
			>
				{post.data.title}
			</a>
		</div>
	);
}
