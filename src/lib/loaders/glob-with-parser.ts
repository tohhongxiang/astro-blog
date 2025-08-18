import { type ParseDataOptions, glob } from "astro/loaders";

type Parser = <TData extends Record<string, unknown>>(
	options: ParseDataOptions<TData>,
) => Promise<ParseDataOptions<TData>>;
type GlobWithParserOptions = Parameters<typeof glob>[0] & {
	parser: Parser;
};

export default function globWithParser({
	parser = async (entry) => entry,
	...globOptions
}: GlobWithParserOptions) {
	const loader = glob(globOptions);
	const originalLoad = loader.load;

	loader.load = async ({ parseData, ...rest }) =>
		originalLoad({
			parseData: async (entry) => parseData(await parser(entry)),
			...rest,
		});

	return loader;
}
