// PIL's Image.resize with LANCZOS on 8-bit pictures, to the byte: the
// filter's taps as resample.c computes them, turned into 22-bit fixed point,
// a horizontal pass rounded to 8 bits, then a vertical one. Models trained on
// PIL-resized inputs (Pixal3D, TRELLIS) see the same pixels.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 1 << PRECISION_BITS;

function sinc(x) {
	if (x === 0) return 1;
	x *= Math.PI;
	return Math.sin(x) / x;
}
const lanczos = (x) => (x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0);

// Per output pixel: the first input pixel and the fixed-point weights.
function taps(inSize, outSize) {
	const scale = inSize / outSize, filterscale = Math.max(scale, 1), support = 3 * filterscale;
	const out = [];
	for (let o = 0; o < outSize; o++) {
		const center = (o + 0.5) * scale, ss = 1 / filterscale;
		const lo = Math.max(Math.trunc(center - support + 0.5), 0);
		const hi = Math.min(Math.trunc(center + support + 0.5), inSize);
		const w = new Float64Array(hi - lo);
		let sum = 0;
		for (let x = 0; x < w.length; x++) sum += (w[x] = lanczos((x + lo - center + 0.5) * ss));
		const k = new Int32Array(w.length);
		for (let x = 0; x < w.length; x++) {
			const v = sum !== 0 ? w[x] / sum : w[x];
			k[x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
		}
		out.push({ lo, k });
	}
	return out;
}

// (sum + half) >> 22, clamped to a byte. The sum fits a float64 exactly.
function clip8(sum) {
	const v = Math.floor((sum + (1 << (PRECISION_BITS - 1))) / ONE);
	return v < 0 ? 0 : v > 255 ? 255 : v;
}

// Interleaved 8-bit pixels [h * w * channels] to [oh * ow * channels].
export function resizeLanczos(pixels, w, h, channels, ow, oh) {
	let src = pixels, sw = w;
	if (ow !== w) {
		const tx = taps(w, ow), mid = new Uint8Array(h * ow * channels);
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < ow; x++) {
				const { lo, k } = tx[x];
				for (let c = 0; c < channels; c++) {
					let s = 0;
					for (let i = 0; i < k.length; i++) s += src[(y * w + lo + i) * channels + c] * k[i];
					mid[(y * ow + x) * channels + c] = clip8(s);
				}
			}
		}
		src = mid;
		sw = ow;
	}
	if (oh === h) return src === pixels ? Uint8Array.from(pixels) : src;
	const ty = taps(h, oh), out = new Uint8Array(oh * sw * channels);
	for (let y = 0; y < oh; y++) {
		const { lo, k } = ty[y];
		for (let x = 0; x < sw; x++) {
			for (let c = 0; c < channels; c++) {
				let s = 0;
				for (let i = 0; i < k.length; i++) s += src[((lo + i) * sw + x) * channels + c] * k[i];
				out[(y * sw + x) * channels + c] = clip8(s);
			}
		}
	}
	return out;
}

// Python's round(): halves go to the even neighbour.
export function roundHalfEven(x) {
	const r = Math.round(x);
	return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
