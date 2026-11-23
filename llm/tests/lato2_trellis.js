// TRELLIS's sparse-structure flow and decoder (LATO.2's stage 0) against the
// torch reference: one velocity with and without the image, the whole
// sampler from the reference's noise, and the decoder's occupancy.
//
//   llm-runner tests/lato2_trellis.js golden=build/lato2/golden/example0 weights="/path/lato2" [linear=q8]
import { llm, f32 } from '/core/llm.js';
import { submit } from '/core/gpu.js';
import { SparseStructureFlow, SparseStructureDecoder } from '/plugins/lato2/trellis.js';
import { upload, readFloats } from '/plugins/lato2/nn.js';
import { openSafetensors } from '/core/weights/torch.js';

const cfg = globalThis.__llm_config;
const dino = llm.open(`${cfg.golden}/dino.safetensors`);
const ss = openSafetensors(`${cfg.golden}/ss_flow.safetensors`);
const dec = llm.open(`${cfg.golden}/ss_dec.safetensors`);

// [C, n] channel-first to rows [n, C] and back.
const toRows = (a, C) => { const n = a.length / C, o = new Float32Array(a.length); for (let c = 0; c < C; c++) for (let i = 0; i < n; i++) o[i * C + c] = a[c * n + i]; return o; };
function compare(name, got, want) {
	let worst = 0, sum = 0, mag = 0;
	for (let i = 0; i < want.length; i++) { const d = Math.abs(got[i] - want[i]); worst = Math.max(worst, d); sum += d; mag += Math.abs(want[i]); }
	llm.print(`${name}: worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)} against mean |x| ${(mag / want.length).toFixed(3)}`);
}

const flow = new SparseStructureFlow(`${cfg.weights}/trellis`, { linear: cfg.linear ?? 'f32' });
await flow.load();
const cond = upload(dino.floats('cond')), uncond = f32(1374 * 1024);
const ctx = flow.context(cond, 1374), nctx = flow.context(uncond, 1374);
const x = upload(toRows(ss.floats('noise'), 8)), v = f32(4096 * 8);
const t0 = ss.floats('t0')[0] / 1000;
flow.forward(x, t0, ctx, v);
compare(`velocity at t ${t0.toFixed(3)}, image`, readFloats(v, 4096 * 8), toRows(ss.floats('v_cond0'), 8));
flow.forward(x, t0, nctx, v);
compare(`velocity at t ${t0.toFixed(3)}, none`, readFloats(v, 4096 * 8), toRows(ss.floats('v_uncond0'), 8));
for (const c of [...ctx, ...nctx]) c.dispose();
await flow.sample(x, cond, uncond, 1374, { steps: ss.ints('steps')[0] });
const z = readFloats(x, 4096 * 8);
compare('sampled latent', z, toRows(ss.floats('z'), 8));
flow.dispose();

const decoder = new SparseStructureDecoder(`${cfg.weights}/trellis`);
decoder.load();
const t1 = llm.now();
const { logits, R } = decoder.run(upload(toRows(dec.floats('z'), 8)), 16);
const got = readFloats(logits, R ** 3), want = dec.floats('logits');
llm.print(`decoder: ${R}^3 in ${llm.since(t1)}`);
compare('occupancy logits (reference latent)', got, want);
let both = 0, onlyRef = 0, onlyOurs = 0;
for (let i = 0; i < want.length; i++) { const a = got[i] > 0, b = want[i] > 0; if (a && b) both++; else if (b) onlyRef++; else if (a) onlyOurs++; }
llm.print(`occupied voxels: ${both} both, ${onlyRef} only the reference's, ${onlyOurs} only ours`);
decoder.dispose();
