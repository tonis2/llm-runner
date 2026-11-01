<*
 The joint-finder trainer's host loops (`llm/plugins/creature/train.js`) in
 C3: QuickJS runs them slowly, and the GPU waits while it does. Each is a
 line-for-line port of its JS function, in the same order and precision, so
 the two agree (to rounding for the float64 one, byte for byte for the
 pictures); train.js keeps the JS as the reference and the fallback.

 `orthonormalRows(Z, n, D)`: Z a Float64Array [n, D], made orthonormal in
 place (train.js orthonormalRows: Cholesky QR twice, rows normalised first).

 `imageComposite(render, channels, backdrop, out, gains, lift, grain, noise,
 noiseAt)`: train.js composite's blend. `gains` a Float64Array of the three
 channels' gains, `noise` a Float32Array read from `noiseAt` when `grain` > 0.
 `out` (a Uint8ClampedArray the backdrop's size) is written.

 `imageCropNearest(pixels, width, height, channels, x0, y0, side, flip, size,
 out)`: train.js Backdrops.photo's crop, a size x size x 3 Uint8Array.

 `imageBackdrop(kind, params, size, out)`: train.js Backdrops.made's loops for
 a background of `kind` (0 to 4), its random draws in `params` (a
 Float64Array, laid out as made() lays them), into the size x size x 3
 Uint8Array `out`.
*>
module llm::host;

import three;
import qjs;
import std::math;

macro double[] @f64s(js, value) => (double[])((double*)js.bytes(value).ptr)[:js.bytes(value).len / 8];
macro float[] @f32s(js, value) => (float[])((float*)js.bytes(value).ptr)[:js.bytes(value).len / 4];

<*
 Rows of Z [n, D] made orthonormal, spanning the same space: train.js
 orthonormalRows exactly.
*>
fn void orthonormal_rows(double[] z, usz n, usz d)
{
	double[] l = mem::new_array(double, (sz)(n * n));
	defer mem::free(l);
	for (int pass = 0; pass < 2; pass++)
	{
		for (usz i = 0; i < n; i++)
		{
			double* zi = &z[i * d];
			double s = 0;
			for (usz k = 0; k < d; k++) s += zi[k] * zi[k];
			double inv = s > 0 ? 1 / math::sqrt(s) : 0;
			for (usz k = 0; k < d; k++) zi[k] *= inv;
		}
		double trace = 0;
		for (usz i = 0; i < n; i++)
		{
			double* zi = &z[i * d];
			for (usz j = 0; j <= i; j++)
			{
				double* zj = &z[j * d];
				double s = 0;
				for (usz k = 0; k < d; k++) s += zi[k] * zj[k];
				l[i * n + j] = s;
			}
			trace += l[i * n + i];
		}
		double eps = (1e-12 * trace) / n + 1e-300;
		for (usz i = 0; i < n; i++)
		{
			for (usz j = 0; j <= i; j++)
			{
				double s = l[i * n + j] + (i == j ? eps : 0);
				for (usz k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
				l[i * n + j] = i == j ? math::sqrt(math::max(s, eps)) : s / l[j * n + j];
			}
		}
		for (usz i = 0; i < n; i++)
		{
			double* zi = &z[i * d];
			for (usz k = 0; k < i; k++)
			{
				double c = l[i * n + k];
				double* zk = &z[k * d];
				for (usz e = 0; e < d; e++) zi[e] -= c * zk[e];
			}
			double inv = 1 / l[i * n + i];
			for (usz e = 0; e < d; e++) zi[e] *= inv;
		}
	}
}

<*
 A number stored into a Uint8ClampedArray: NaN to 0, clamped to 0..255, then
 rounded half to even (ECMAScript ToUint8Clamp).
*>
fn char to_uint8_clamp(double x) @inline
{
	if (!(x > 0)) return 0;
	if (x >= 255) return 255;
	double f = math::floor(x);
	double r = x - f;
	if (r > 0.5) return (char)(f + 1);
	if (r < 0.5) return (char)f;
	return ((long)f % 2 == 0) ? (char)f : (char)(f + 1);
}

<*
 train.js composite's blend: the render over the backdrop through each
 channel's table (v * gain + lift, kept in float32 as Float32Array.from keeps
 it), then the grain.
*>
fn void composite(char[] src, usz channels, char[] backdrop, char[] out, double[3] gains, double lift, double grain, float[] noise, usz noise_at)
{
	float[256][3] table;
	for (usz c = 0; c < 3; c++)
	{
		for (usz v = 0; v < 256; v++) table[c][v] = (float)(v * gains[c] + lift);
	}
	out[..] = backdrop[..];
	usz pixels = out.len / 3;
	for (usz i = 0; i < pixels; i++)
	{
		usz s = i * channels;
		usz d = i * 3;
		int a = channels == 4 ? src[s + 3] : 255;
		if (a == 0) continue;
		if (a == 255)
		{
			for (usz c = 0; c < 3; c++) out[d + c] = to_uint8_clamp(table[c][src[s + c]]);
			continue;
		}
		double t = a / 255.0;
		double u = 1 - t;
		for (usz c = 0; c < 3; c++) out[d + c] = to_uint8_clamp((double)table[c][src[s + c]] * t + backdrop[d + c] * u);
	}
	if (grain > 0)
	{
		usz n = noise_at;
		for (usz d = 0; d < out.len; d++) out[d] = to_uint8_clamp(out[d] + grain * noise[n++]);
	}
}

<*
 train.js Backdrops.photo's crop: a side x side square at (x0, y0), read
 nearest-pixel onto size x size, mirrored with `flip`.
*>
fn void crop_nearest(char[] pixels, usz width, usz height, usz channels, double x0, double y0, double side, bool flip, usz size, char[] out)
{
	for (usz y = 0; y < size; y++)
	{
		usz sy = (usz)math::min((double)(height - 1), math::floor(y0 + (y * side) / size));
		for (usz x = 0; x < size; x++)
		{
			usz sx = (usz)math::min((double)(width - 1), math::floor(x0 + (x * side) / size));
			if (flip) sx = width - 1 - sx;
			usz s = (sy * width + sx) * channels;
			usz d = (y * size + x) * 3;
			out[d] = pixels[s];
			out[d + 1] = pixels[s + 1];
			out[d + 2] = pixels[s + 2];
		}
	}
}

fn void js_orthonormal_rows(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out)
{
	qjs::Context js = qjs::wrap(ctx);
	if (argc < 3) { *out = js.raise("orthonormalRows(Z, n, D)"); return; }
	double[] z = @f64s(js, args[0]);
	long n = three::arg_int(&js, argc, args, 1);
	long d = three::arg_int(&js, argc, args, 2);
	if (n <= 0 || d <= 0 || z.len < (usz)(n * d))
	{
		*out = js.raise("orthonormalRows: Z must be a Float64Array of n x D");
		return;
	}
	orthonormal_rows(z, (usz)n, (usz)d);
}

fn void js_image_composite(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out)
{
	qjs::Context js = qjs::wrap(ctx);
	if (argc < 9) { *out = js.raise("imageComposite(render, channels, backdrop, out, gains, lift, grain, noise, noiseAt)"); return; }
	char[] src = js.bytes(args[0]);
	long channels = three::arg_int(&js, argc, args, 1);
	char[] backdrop = js.bytes(args[2]);
	char[] dst = js.bytes(args[3]);
	double[] gains = @f64s(js, args[4]);
	double lift = js.number(args[5]);
	double grain = js.number(args[6]);
	float[] noise = @f32s(js, args[7]);
	long noise_at = three::arg_int(&js, argc, args, 8);
	usz pixels = dst.len / 3;
	if ((channels != 3 && channels != 4) || dst.len % 3 != 0 || backdrop.len != dst.len || src.len < pixels * (usz)channels || gains.len < 3
		|| (grain > 0 && (noise_at < 0 || (usz)noise_at + dst.len > noise.len)))
	{
		*out = js.raise("imageComposite: the render, backdrop, out, gains or noise do not match");
		return;
	}
	composite(src, (usz)channels, backdrop, dst, { gains[0], gains[1], gains[2] }, lift, grain, noise, (usz)noise_at);
}

fn void js_image_crop_nearest(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out)
{
	qjs::Context js = qjs::wrap(ctx);
	if (argc < 10) { *out = js.raise("imageCropNearest(pixels, width, height, channels, x0, y0, side, flip, size, out)"); return; }
	char[] pixels = js.bytes(args[0]);
	long width = three::arg_int(&js, argc, args, 1);
	long height = three::arg_int(&js, argc, args, 2);
	long channels = three::arg_int(&js, argc, args, 3);
	double x0 = js.number(args[4]);
	double y0 = js.number(args[5]);
	double side = js.number(args[6]);
	bool flip = js.number(args[7]) != 0;
	long size = three::arg_int(&js, argc, args, 8);
	char[] dst = js.bytes(args[9]);
	if (width <= 0 || height <= 0 || channels < 3 || size <= 0 || pixels.len < (usz)(width * height * channels)
		|| dst.len < (usz)(size * size * 3) || !(x0 >= 0) || !(y0 >= 0))
	{
		*out = js.raise("imageCropNearest: the image or the crop do not match");
		return;
	}
	crop_nearest(pixels, (usz)width, (usz)height, (usz)channels, x0, y0, side, flip, (usz)size, dst);
}

faultdef BACKDROP_BAD;

<*
 A number stored into a Uint8Array: truncated toward zero, then modulo 256
 (ECMAScript ToUint8).
*>
fn char to_uint8(double x) @inline
{
	if (!(x == x) || x > 1e18 || x < -1e18) return 0;
	return (char)((long)x & 255);
}

<*
 train.js Backdrops.made's loops, the draws given: kind 0 [a(3), b(3)], 1
 [sky(3), ground(3), horizon], 2 [a(3), b(3), tile], 3 [g, grid((g+1)^2 x 3)],
 4 [tone, tint(3), horizon, floor].
*>
fn void? backdrop(long kind, double[] p, usz size, char[] px)
{
	double s = size;
	switch (kind)
	{
		case 0:
			if (p.len < 6) return BACKDROP_BAD~;
			for (usz y = 0; y < size; y++) for (usz x = 0; x < size; x++) for (usz c = 0; c < 3; c++)
			{
				px[(y * size + x) * 3 + c] = to_uint8(p[c] + ((p[3 + c] - p[c]) * y) / s);
			}
		case 1:
			if (p.len < 7) return BACKDROP_BAD~;
			double horizon = p[6];
			for (usz y = 0; y < size; y++)
			{
				bool up = y < horizon;
				double shade = up ? 1 - (0.3 * y) / s : 0.7 + (0.3 * y) / s;
				for (usz x = 0; x < size; x++) for (usz c = 0; c < 3; c++)
				{
					px[(y * size + x) * 3 + c] = to_uint8(math::min(255.0, p[(up ? 0u : 3u) + c] * shade));
				}
			}
		case 2:
			if (p.len < 7 || p[6] < 1) return BACKDROP_BAD~;
			usz t = (usz)p[6];
			for (usz y = 0; y < size; y++) for (usz x = 0; x < size; x++)
			{
				usz from = ((x / t) + (y / t)) % 2 == 1 ? 0 : 3;
				for (usz c = 0; c < 3; c++) px[(y * size + x) * 3 + c] = to_uint8(p[from + c]);
			}
		case 4:
			if (p.len < 6) return BACKDROP_BAD~;
			double tone = p[0];
			double horizon = p[4];
			double floor = p[5];
			for (usz y = 0; y < size; y++)
			{
				double shade = y < horizon ? 1 - (0.08 * y) / horizon : floor * (0.9 + (0.1 * (y - horizon)) / math::max(1.0, s - horizon));
				for (usz x = 0; x < size; x++) for (usz c = 0; c < 3; c++)
				{
					px[(y * size + x) * 3 + c] = to_uint8(math::max(0.0, math::min(255.0, (tone + p[1 + c]) * shade)));
				}
			}
		case 3:
			if (p.len < 1 || p[0] < 1) return BACKDROP_BAD~;
			usz g = (usz)p[0];
			if (p.len < 1 + (g + 1) * (g + 1) * 3) return BACKDROP_BAD~;
			double[] grid = p[1..];
			for (usz y = 0; y < size; y++)
			{
				double fy = ((double)y / s) * g;
				double y0 = math::floor(fy);
				double ty = fy - y0;
				for (usz x = 0; x < size; x++)
				{
					double fx = ((double)x / s) * g;
					double x0 = math::floor(fx);
					double tx = fx - x0;
					usz q00 = ((usz)y0 * (g + 1) + (usz)x0) * 3;
					usz q10 = q00 + 3;
					usz q01 = q00 + (g + 1) * 3;
					usz q11 = q01 + 3;
					for (usz c = 0; c < 3; c++)
					{
						px[(y * size + x) * 3 + c] = to_uint8((grid[q00 + c] * (1 - tx) + grid[q10 + c] * tx) * (1 - ty)
							+ (grid[q01 + c] * (1 - tx) + grid[q11 + c] * tx) * ty);
					}
				}
			}
		default:
			return BACKDROP_BAD~;
	}
}

fn void js_image_backdrop(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out)
{
	qjs::Context js = qjs::wrap(ctx);
	if (argc < 4) { *out = js.raise("imageBackdrop(kind, params, size, out)"); return; }
	long kind = three::arg_int(&js, argc, args, 0);
	double[] params = @f64s(js, args[1]);
	long size = three::arg_int(&js, argc, args, 2);
	char[] dst = js.bytes(args[3]);
	if (size <= 0 || dst.len < (usz)(size * size * 3))
	{
		*out = js.raise("imageBackdrop: out must hold size x size x 3 bytes");
		return;
	}
	if (catch backdrop(kind, params, (usz)size, dst)) *out = js.raise("imageBackdrop: the kind or its draws do not match Backdrops.made's");
}
