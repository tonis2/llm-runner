// Smoothing along frames, shared by the plugins that write clips.

// Savitzky-Golay weights (quadratic) for a window of `size` frames, odd.
function savitzkyGolay(size) {
	const m = (size - 1) >> 1;
	const d = (2 * m - 1) * (2 * m + 1) * (2 * m + 3);
	return Array.from({ length: 2 * m + 1 }, (_, k) => {
		const i = k - m;
		return (3 * (3 * m * m + 3 * m - 1 - 5 * i * i)) / d;
	});
}

// `data` [frames, width] smoothed along frames. Past either end the frames
// are reflected through the end one (2 x[0] - x[k]), which keeps a steady
// drift steady where a mirror would bend it.
export function smoothRows(data, frames, width, size) {
	if (size < 3 || frames < 3) return;
	size = Math.min(size | 1, frames % 2 ? frames : frames - 1);
	if (size < 3) return;
	const w = savitzkyGolay(size), m = (size - 1) >> 1;
	const src = Float64Array.from(data);
	const value = (f, c) => {
		if (f < 0) return 2 * src[c] - src[-f * width + c];
		if (f >= frames) return 2 * src[(frames - 1) * width + c] - src[(2 * (frames - 1) - f) * width + c];
		return src[f * width + c];
	};
	for (let f = 0; f < frames; f++) {
		for (let c = 0; c < width; c++) {
			let s = 0;
			for (let k = -m; k <= m; k++) s += w[k + m] * value(f + k, c);
			data[f * width + c] = s;
		}
	}
}
