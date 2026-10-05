// Two LoRAs on Qwen-Image block 0, merged and unmerged, against the host:
// the change each projection's output takes from the pair, on sampled rows.
// `llm-runner tests/lora_stack.js dit=<gguf> lora1=<st> lora2=<st> [merge=false]`
import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit } from '../lib/gpu.js';
import { mergeLoras } from '../qwenimage/lora.js';

const args = globalThis.__llm_config ?? {};
const merge = String(args.merge) !== 'false';
const m = llm.open(args.dit);
const p = m.has('model.diffusion_model.img_in.weight') ? 'model.diffusion_model.' : '';
const b = 'transformer_blocks.0.';
const names = { q: 'attn.to_q', k: 'attn.to_k', v: 'attn.to_v', out: 'attn.to_out.0', gateUp: 'img_mlp.gate_up', down: 'img_mlp.out' };
const loras = [args.lora1, args.lora2].filter(Boolean).map((path) => ({ path, strength: 1 }));
const seq = 64;
let seed = 3;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5) * 2;

function run(withLoras) {
	const blk = {};
	for (const [k, n] of Object.entries(names)) blk[k] = m.upload(`${p}${b}${n}.weight`, 'raw');
	const ffn = blk.gateUp.shape[1] / 2;
	const dit = { nLayers: 1, blocks: [blk], ffn };
	if (withLoras) mergeLoras(dit, loras, merge);
	const out = {};
	// The block's order: Q, K, V off one input; gate and up off one.
	const D = blk.q.shape[0];
	const xs = {};
	const mk = (n) => { const h = Float32Array.from({ length: seq * n }, rnd); const t = f32(h.length); t.buffer.write(h); return [h, t]; };
	seed = 3;
	const [hx, x] = mk(D), [ha, xa] = mk(D), [hg, xg] = mk(ffn);
	const ys = {};
	const y = (k, rows) => (ys[k] = f32(seq * rows));
	op.matmul(blk.q, x, y('q', D), D, D, seq, 0, true);
	op.matmul(blk.k, x, y('k', D), D, D, seq, 0, true);
	op.matmul(blk.v, x, y('v', D), D, D, seq);
	op.matmul(blk.out, xa, y('out', D), D, D, seq);
	op.matmul(blk.gateUp, x, y('gate', ffn), ffn, D, seq, 0, true);
	op.matmul(blk.gateUp, x, y('up', ffn), ffn, D, seq, ffn);
	op.matmul(blk.down, xg, y('down', D), D, ffn, seq);
	submit();
	for (const [k, t] of Object.entries(ys)) { out[k] = new Float32Array(t.buffer.readBytes().buffer); t.dispose(); }
	for (const t of Object.values(blk)) { for (const ad of t.adapters ?? []) ad.dispose(); t.dispose(); }
	return { out, x: hx, xa: ha, xg: hg, D, ffn };
}

const base = run(false);
const lora = run(true);
const { D, ffn } = base;

// Host: sum over the LoRAs of scale x A^T B^T, for `rows` of a site.
function hostDelta(site, x, inDim, rows, rowOffset) {
	const d = rows.map(() => new Float64Array(seq));
	for (const { path } of loras) {
		const f = llm.open(path);
		const find = (st) => ['diffusion_model.', 'transformer.', ''].map((x) => `${x}${b}${st}`).find((x) => [...f.tensors.keys()].some((k) => k.startsWith(`${x}.lora_A`)));
		let pre = find(site), off = rowOffset;
		// A file that names the gate and up halves on their own.
		if (!pre && site === 'img_mlp.gate_up') { pre = find(rowOffset === 0 ? 'img_mlp.gate_layer' : 'img_mlp.proj'); off = 0; }
		const tail = f.has(`${pre}.lora_A.weight`) ? '.weight' : '.default.weight';
		const A = f.floats(`${pre}.lora_A${tail}`), B = f.floats(`${pre}.lora_B${tail}`);
		const rank = A.length / inDim;
		const alphaT = f.has(`${pre}.alpha`) ? f.floats(`${pre}.alpha`)[0] : rank;
		const scale = alphaT / rank;
		f.close();
		const t = new Float64Array(seq * rank);
		for (let s = 0; s < seq; s++) for (let r = 0; r < rank; r++) {
			let acc = 0;
			for (let i = 0; i < inDim; i++) acc += x[s * inDim + i] * A[r * inDim + i];
			t[s * rank + r] = acc;
		}
		rows.forEach((row, j) => {
			for (let s = 0; s < seq; s++) {
				let acc = 0;
				for (let r = 0; r < rank; r++) acc += t[s * rank + r] * B[(off + row) * rank + r];
				d[j][s] += scale * acc;
			}
		});
	}
	return d;
}

const cases = [
	['q', 'attn.to_q', base.x, D, D, 0], ['k', 'attn.to_k', base.x, D, D, 0], ['v', 'attn.to_v', base.x, D, D, 0],
	['out', 'attn.to_out.0', base.xa, D, D, 0],
	['gate', 'img_mlp.gate_up', base.x, D, ffn, 0], ['up', 'img_mlp.gate_up', base.x, D, ffn, ffn],
	['down', 'img_mlp.out', base.xg, ffn, D, 0],
];
for (const [key, site, x, inDim, outDim, rowOffset] of cases) {
	const rows = Array.from({ length: 24 }, (_, i) => Math.floor(i * outDim / 24) + 7);
	const want = hostDelta(site, x, inDim, rows, rowOffset);
	let err = 0, mag = 0;
	rows.forEach((row, j) => {
		for (let s = 0; s < seq; s++) {
			const got = lora.out[key][s * outDim + row] - base.out[key][s * outDim + row];
			err += (got - want[j][s]) ** 2;
			mag += want[j][s] ** 2;
		}
	});
	llm.print(`${merge ? 'merged  ' : 'unmerged'} ${key.padEnd(5)} |delta| ${Math.sqrt(mag / rows.length / seq).toExponential(2)}  rel err ${Math.sqrt(err / mag).toFixed(4)}`);
}
