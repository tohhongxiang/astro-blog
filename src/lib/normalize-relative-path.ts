import path from "node:path";

export default function normalizeRelativePath(
	from: string,
	to: string,
): string {
	return path.relative(from, to).split(path.sep).join("/");
}
