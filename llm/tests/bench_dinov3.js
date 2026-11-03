// SAM 3D Body's backbone at full size (ViT-H+/16, 32 blocks, 1029 tokens at
// 512x512) on random weights: how long one pass takes on this GPU, with no
// download. `llm-runner tests/bench_dinov3.js [linear=q8|f32] [passes=3] [depth=32]`;
// `profile=true` adds the time by kernel.
import { llm, f32, Tensor, GGML } from '/core/llm.js';
import { DinoV3 } from '/core/models/dinov3.js';

const cfg = globalThis.__llm_config;
const linear = cfg.linear ?? 'q8';
const passes = cfg.passes ?? 3;
const D = 1280, F = 5120, DEPTH = cfg.depth ?? 32;
const C = three.compute;

// PyTorch-order shapes of the tensors DinoV3 reads.
const shapes = new Map([
	['cls_token', [1, 1, D]], ['storage_tokens', [1, 4, D]],
	['patch_embed.proj.weight', [D, 3, 16, 16]], ['patch_embed.proj.bias', [D]],
	['norm.weight', [D]], ['norm.bias', [D]], ['rope_embed.periods', [16]],
]);
for (let l = 0; l < DEPTH; l++) {
	const b = `blocks.${l}.`;
	for (const [k, s] of [['norm1.weight', [D]], ['norm1.bias', [D]], ['attn.qkv.weight', [3 * D, D]], ['attn.qkv.bias', [3 * D]],
		['attn.proj.weight', [D, D]], ['attn.proj.bias', [D]], ['ls1.gamma', [D]], ['norm2.weight', [D]], ['norm2.bias', [D]],
		['mlp.w1.weight', [F, D]], ['mlp.w1.bias', [F]], ['mlp.w2.weight', [F, D]], ['mlp.w2.bias', [F]],
		['mlp.w3.weight', [D, F]], ['mlp.w3.bias', [D]], ['ls2.gamma', [D]]]) shapes.set(b + k, s);
}
let seed = 5;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5);
const count = (s) => s.reduce((a, b) => a * b, 1);
const small = (name, n) => {
	if (name.endsWith('periods')) return Float32Array.from({ length: n }, (_, i) => Math.pow(100, (2 * i) / 32));
	if (name.endsWith('gamma')) return new Float32Array(n).fill(1e-2);
	if (/norm\d?\.weight$/.test(name)) return new Float32Array(n).fill(1);
	return Float32Array.from({ length: Math.min(n, 4096) }, () => rnd() * 0.02);
};
// A model-like object: random values, uploaded without a file.
const fake = {
	has: (name) => shapes.has(name.replace('backbone.encoder.', '')),
	shape: (name) => shapes.get(name.replace('backbone.encoder.', '')).slice().reverse(),
	floats(name) {
		const n = count(shapes.get(name.replace('backbone.encoder.', '')));
		const v = small(name, n);
		if (v.length === n) return v;
		const out = new Float32Array(n);
		for (let i = 0; i < n; i += v.length) out.set(v.subarray(0, Math.min(v.length, n - i)), i);
		return out;
	},
	upload(name, as) {
		const s = shapes.get(name.replace('backbone.encoder.', '')), n = count(s);
		if (as === 'q8') {
			// Q8_0 blocks: an f16 scale (0.0002) and 32 int8s.
			const blocks = n / 32, bytes = new Uint8Array(blocks * 34);
			for (let b = 0; b < blocks; b++) {
				bytes[b * 34] = 0x8e; bytes[b * 34 + 1] = 0x0a;
				for (let i = 0; i < 32; i++) bytes[b * 34 + 2 + i] = (b * 7 + i * 13) & 0xff;
			}
			const buf = C.f32(Math.ceil(bytes.length / 4));
			buf.write(new Float32Array(bytes.buffer, 0, Math.floor(bytes.length / 4)));
			return new Tensor(buf, GGML.Q8_0, s.slice().reverse(), name);
		}
		const t = f32(n, s.slice().reverse(), name);
		t.buffer.write(this.floats(name));
		return t;
	},
};

const enc = new DinoV3(fake, { linear });
await enc.load();
const pixels = Float32Array.from({ length: 3 * 512 * 512 }, () => rnd() * 2);
const times = [];
for (let i = 0; i < passes; i++) {
	const t0 = llm.now();
	const feat = await enc.encode(pixels, 512, 512);
	feat.buffer.readBytes();
	times.push(llm.now() - t0);
	feat.dispose();
}
enc.dispose();
(await import('/core/gpu.js')).profileReport('dinov3');
const best = Math.min(...times);
const flops = DEPTH * 1029 * 2 * (4 * D * D + 3 * D * F) + DEPTH * 4 * 1029 * 1029 * D;
llm.print(`DINOv3 ViT-H+ ${linear}, ${DEPTH} blocks at 512x512: best ${(best / 1000).toFixed(2)} s a pass (${times.map((t) => (t / 1000).toFixed(2)).join(', ')}), ${(flops / best / 1e9).toFixed(2)} TFLOPS`);
