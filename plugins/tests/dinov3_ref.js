// sam3dbody/dinov3.js against a float64 host forward written from dinov3's
// PyTorch code, on a small random model, so the layout (prefix tokens, the RoPE
// on patch rows only, rows against columns, Q/K/V thirds, the masked K bias)
// is checked without SAM 3D Body's weights.
//
//   llm-runner tests/dinov3_ref.js [linear=q8]
//
// The weights are written as a bf16 safetensors file first, the checkpoint's
// own type, and the reference reads the same rounded values.
import { llm } from '../lib/llm.js';
import { DinoV3, PATCH } from '../lib/dinov3.js';

const cfg = globalThis.__llm_config;
const linear = cfg.linear ?? 'f32';
const D = 128, HEADS = 2, HD = 64, F = 256, DEPTH = 2, STORE = 4, GH = 4, GW = 3;
const H = GH * PATCH, W = GW * PATCH, N = GH * GW, PRE = 1 + STORE, n = PRE + N;

let seed = 11;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5);
const bf16 = (v) => {
	const f = new Float32Array([v]), u = new Uint32Array(f.buffer);
	u[0] = (u[0] + 0x7fff + ((u[0] >> 16) & 1)) & 0xffff0000;
	return f[0];
};

// name -> { shape (PyTorch order), data (bf16-rounded floats) }
const T = {};
const put = (name, shape, make) => {
	const count = shape.reduce((a, b) => a * b, 1);
	T[`backbone.encoder.${name}`] = { shape, data: Float64Array.from({ length: count }, (_, i) => bf16(make(i))) };
};
put('cls_token', [1, 1, D], () => rnd());
put('storage_tokens', [1, STORE, D], () => rnd());
put('patch_embed.proj.weight', [D, 3, PATCH, PATCH], () => rnd() * 0.1);
put('patch_embed.proj.bias', [D], () => rnd() * 0.1);
put('norm.weight', [D], () => 1 + rnd() * 0.2);
put('norm.bias', [D], () => rnd() * 0.1);
put('rope_embed.periods', [HD / 4], (i) => Math.pow(100, (2 * i) / (HD / 2)));
for (let l = 0; l < DEPTH; l++) {
	const b = `blocks.${l}.`;
	put(`${b}norm1.weight`, [D], () => 1 + rnd() * 0.2);
	put(`${b}norm1.bias`, [D], () => rnd() * 0.1);
	put(`${b}attn.qkv.weight`, [3 * D, D], () => rnd() * 0.2);
	put(`${b}attn.qkv.bias`, [3 * D], () => rnd() * 0.2);
	put(`${b}attn.qkv.bias_mask`, [3 * D], (i) => (i >= D && i < 2 * D ? 0 : 1));
	put(`${b}attn.proj.weight`, [D, D], () => rnd() * 0.2);
	put(`${b}attn.proj.bias`, [D], () => rnd() * 0.1);
	put(`${b}ls1.gamma`, [D], () => rnd());
	put(`${b}norm2.weight`, [D], () => 1 + rnd() * 0.2);
	put(`${b}norm2.bias`, [D], () => rnd() * 0.1);
	put(`${b}mlp.w1.weight`, [F, D], () => rnd() * 0.2);
	put(`${b}mlp.w1.bias`, [F], () => rnd() * 0.1);
	put(`${b}mlp.w2.weight`, [F, D], () => rnd() * 0.2);
	put(`${b}mlp.w2.bias`, [F], () => rnd() * 0.1);
	put(`${b}mlp.w3.weight`, [D, F], () => rnd() * 0.2);
	put(`${b}mlp.w3.bias`, [D], () => rnd() * 0.1);
	put(`${b}ls2.gamma`, [D], () => rnd());
}

// The safetensors file: an 8-byte header length, the JSON header, the data.
function writeSafetensors(path) {
	const header = {};
	let at = 0;
	for (const [name, t] of Object.entries(T)) {
		header[name] = { dtype: 'BF16', shape: t.shape, data_offsets: [at, at + t.data.length * 2] };
		at += t.data.length * 2;
	}
	let json = JSON.stringify(header);
	while ((8 + json.length) % 8) json += ' ';
	const bytes = new Uint8Array(8 + json.length + at);
	const dv = new DataView(bytes.buffer);
	dv.setUint32(0, json.length, true);
	for (let i = 0; i < json.length; i++) bytes[8 + i] = json.charCodeAt(i);
	let p = 8 + json.length;
	const f = new Float32Array(1), u = new Uint32Array(f.buffer);
	for (const t of Object.values(T)) {
		for (const v of t.data) { f[0] = v; dv.setUint16(p, u[0] >>> 16, true); p += 2; }
	}
	llm.writeBytes(path, bytes);
}

const pixels = Float32Array.from({ length: 3 * H * W }, () => rnd() * 2);

// --- the reference, in float64 -----------------------------------------------------

const w = (name) => T[`backbone.encoder.${name}`].data;
function layerNorm(x, g, b) {
	const out = new Float64Array(x.length);
	for (let r = 0; r < x.length / D; r++) {
		let m = 0, v = 0;
		for (let c = 0; c < D; c++) m += x[r * D + c];
		m /= D;
		for (let c = 0; c < D; c++) v += (x[r * D + c] - m) ** 2;
		const s = 1 / Math.sqrt(v / D + 1e-5);
		for (let c = 0; c < D; c++) out[r * D + c] = (x[r * D + c] - m) * s * g[c] + b[c];
	}
	return out;
}
function linear_(x, wt, b, inD, outD, bias = b) {
	const rows = x.length / inD, out = new Float64Array(rows * outD);
	for (let r = 0; r < rows; r++) {
		for (let o = 0; o < outD; o++) {
			let s = bias ? bias[o] : 0;
			for (let i = 0; i < inD; i++) s += x[r * inD + i] * wt[o * inD + i];
			out[r * outD + o] = s;
		}
	}
	return out;
}
function reference() {
	// prepare_tokens_with_masks: the stride-16 conv, flattened row-major, after cls + storage.
	const x = new Float64Array(n * D);
	x.set(w('cls_token'), 0);
	x.set(w('storage_tokens'), D);
	const cw = w('patch_embed.proj.weight'), cb = w('patch_embed.proj.bias');
	for (let py = 0; py < GH; py++) {
		for (let px = 0; px < GW; px++) {
			const row = PRE + py * GW + px;
			for (let o = 0; o < D; o++) {
				let s = cb[o];
				for (let c = 0; c < 3; c++) {
					for (let ky = 0; ky < PATCH; ky++) {
						for (let kx = 0; kx < PATCH; kx++) {
							s += cw[((o * 3 + c) * PATCH + ky) * PATCH + kx] * pixels[c * H * W + (py * PATCH + ky) * W + px * PATCH + kx];
						}
					}
				}
				x[row * D + o] = s;
			}
		}
	}
	// RopePositionEmbedding.forward: sin, cos [HW, HD].
	const periods = w('rope_embed.periods');
	const sin = [], cos = [];
	for (let t = 0; t < N; t++) {
		const ch = 2 * ((Math.floor(t / GW) + 0.5) / GH) - 1, cwid = 2 * ((t % GW + 0.5) / GW) - 1;
		const ang = [];
		for (const c of [ch, cwid]) for (let j = 0; j < HD / 4; j++) ang.push((2 * Math.PI * c) / periods[j]);
		const tiled = [...ang, ...ang];
		sin.push(tiled.map(Math.sin));
		cos.push(tiled.map(Math.cos));
	}
	const rope = (v, t) => {
		const out = new Float64Array(HD);
		for (let j = 0; j < HD; j++) {
			const rot = j < HD / 2 ? -v[j + HD / 2] : v[j - HD / 2];
			out[j] = v[j] * cos[t][j] + rot * sin[t][j];
		}
		return out;
	};
	let h = x;
	for (let l = 0; l < DEPTH; l++) {
		const b = `blocks.${l}.`;
		const bias = w(`${b}attn.qkv.bias`).map((v, i) => v * w(`${b}attn.qkv.bias_mask`)[i]);
		const qkv = linear_(layerNorm(h, w(`${b}norm1.weight`), w(`${b}norm1.bias`)), w(`${b}attn.qkv.weight`), bias, D, 3 * D);
		// [n, 3, heads, hd]
		const at = (tok, which, head) => qkv.subarray(tok * 3 * D + which * D + head * HD, tok * 3 * D + which * D + (head + 1) * HD);
		const attn = new Float64Array(n * D);
		for (let hh = 0; hh < HEADS; hh++) {
			const q = [], k = [], v = [];
			for (let t = 0; t < n; t++) {
				q.push(t < PRE ? at(t, 0, hh) : rope(at(t, 0, hh), t - PRE));
				k.push(t < PRE ? at(t, 1, hh) : rope(at(t, 1, hh), t - PRE));
				v.push(at(t, 2, hh));
			}
			for (let i = 0; i < n; i++) {
				const s = k.map((kk) => kk.reduce((acc, kv, j) => acc + kv * q[i][j], 0) / Math.sqrt(HD));
				const mx = Math.max(...s);
				const e = s.map((z) => Math.exp(z - mx));
				const sum = e.reduce((a2, b2) => a2 + b2, 0);
				for (let j = 0; j < HD; j++) {
					let o = 0;
					for (let t = 0; t < n; t++) o += (e[t] / sum) * v[t][j];
					attn[i * D + hh * HD + j] = o;
				}
			}
		}
		const proj = linear_(attn, w(`${b}attn.proj.weight`), w(`${b}attn.proj.bias`), D, D);
		const h1 = h.map((v, i) => v + w(`${b}ls1.gamma`)[i % D] * proj[i]);
		const x2 = layerNorm(h1, w(`${b}norm2.weight`), w(`${b}norm2.bias`));
		const g1 = linear_(x2, w(`${b}mlp.w1.weight`), w(`${b}mlp.w1.bias`), D, F);
		const g2 = linear_(x2, w(`${b}mlp.w2.weight`), w(`${b}mlp.w2.bias`), D, F);
		const hid = g1.map((v, i) => (v / (1 + Math.exp(-v))) * g2[i]);
		const mlp = linear_(hid, w(`${b}mlp.w3.weight`), w(`${b}mlp.w3.bias`), F, D);
		h = h1.map((v, i) => v + w(`${b}ls2.gamma`)[i % D] * mlp[i]);
	}
	const normed = layerNorm(h, w('norm.weight'), w('norm.bias'));
	// Patch rows, reshaped to [D, gh, gw].
	const out = new Float64Array(D * N);
	for (let t = 0; t < N; t++) for (let c = 0; c < D; c++) out[c * N + t] = normed[(PRE + t) * D + c];
	return out;
}

// --- the port ----------------------------------------------------------------------

llm.makeDir('build');
const path = 'build/dinov3_tiny.safetensors';
writeSafetensors(path);
const m = llm.open(path);
const enc = new DinoV3(m, { linear });
await enc.load();
const feat = await enc.encode(pixels, H, W);
const got = new Float32Array(feat.buffer.readBytes().buffer);
feat.dispose();
enc.dispose();
m.close();

const ref = reference();
let err = 0, scale = 0, dot = 0, na = 0, nb = 0;
for (let i = 0; i < ref.length; i++) {
	err = Math.max(err, Math.abs(got[i] - ref[i]));
	scale = Math.max(scale, Math.abs(ref[i]));
	dot += got[i] * ref[i]; na += got[i] ** 2; nb += ref[i] ** 2;
}
const cosine = dot / Math.sqrt(na * nb);
const limit = linear === 'q8' ? 3e-2 : 1e-4;
const ok = err / scale < limit && cosine > (linear === 'q8' ? 0.999 : 0.999999);
llm.print(`${ok ? 'ok  ' : 'FAIL'} DINOv3 ${linear} weights, [${D}, ${GH}, ${GW}] features: max error ${(err / scale).toExponential(2)} of the largest, cosine ${cosine.toFixed(7)}`);
llm.print(ok ? 'all passed' : '1 FAILED');
