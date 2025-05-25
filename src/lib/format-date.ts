export function formatDate(dateString: string) {
	return new Date(dateString).toISOString().split("T")[0];
}
