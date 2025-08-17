export function generateIdFromRelativePath(rel: string): string {
	const noExt = rel.replace(/\.[^/.]+$/, "");
	const normalized = noExt.replace(/\\/g, "/");

	// Handle index files: egg-drop/index -> egg-drop
	if (normalized.endsWith("/index")) {
		return normalized.replace(/\/index$/, "");
	}

	return normalized;
}
