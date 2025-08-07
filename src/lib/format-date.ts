export function formatDate(dateString: string | Date) {
	if (typeof dateString === "string") {
		dateString = new Date(dateString);
	}

	return dateString.toISOString().split("T")[0];
}
