import { formatDate } from "@/lib/format-date";
import type { Post } from "@/types";

export default function PostItem({ post }: { post: Post }) {
	return (
		<div className="flex flex-row items-baseline gap-4 py-2">
			{post.frontmatter.date && (
				<time
					dateTime={formatDate(post.frontmatter.date)}
					className="text-muted-foreground font-mono"
				>
					{formatDate(post.frontmatter.date)}
				</time>
			)}
			<a href={post.url} className="font-semibold hover:underline">
				{post.frontmatter.title}
			</a>
		</div>
	);
}
