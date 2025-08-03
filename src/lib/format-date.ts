export function formatDate(dateString: string | Date) {
	if (typeof dateString === "string") {
		return new Date(dateString).toISOString().split("T")[0];
	}

	return dateString.toISOString().split("T")[0];
}
