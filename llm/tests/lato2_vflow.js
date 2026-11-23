// LATO.2's V-Flow against the torch reference: the first step's velocities
// with and without the image, then the whole sampler from the same noise.
//
//   llm-runner tests/lato2_vflow.js golden=build/lato2/golden/example0 weights="/path/lato2" [linear=q8]
import { llm, f32 } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { VertexFlow } from '/plugins/lato2/vflow.js';
import { upload, readFloats, ropeTables } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config;
const dino = llm.open(`${cfg.golden}/dino.safetensors`);
const g = openSafetensors(`${cfg.golden}/vflow.safetensors`);
function compare(name, got, want) {
	let worst = 0, sum = 0, mag = 0;
	for (let i = 0; i < want.length; i++) { const d = Math.abs(got[i] - want[i]); worst = Math.max(worst, d); sum += d; mag += Math.abs(want[i]); }
	llm.print(`${name}: worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)} against mean |x| ${(mag / want.length).toFixed(3)}`);
}

const coords = Int32Array.from(g.ints('coords')), n = coords.length / 3;
const flow = new VertexFlow(`${cfg.weights}/vflow.pt`, { linear: cfg.linear ?? 'f32' });
await flow.load();
const cond = upload(dino.floats('cond')), uncond = f32(1374 * 1024);
const ctx = flow.context(cond, 1374), nctx = flow.context(uncond, 1374);
const rope = ropeTables(coords, n, 64);
const x = upload(g.floats('noise')), v = f32(n * 32);
const t0 = g.floats('t0')[0] / 1000, density = g.floats('density')[0];
llm.print(`${n} points, t ${t0}, density ${density} (ours ${flow.densityOf(2000)})`);
let t1 = llm.now();
flow.forward(x, n, rope, t0, density, ctx, v);
compare('velocity, image', readFloats(v, n * 32), g.floats('v_cond0'));
llm.print(`  one pass ${llm.since(t1)}`);
flow.forward(x, n, rope, t0, density, nctx, v);
compare('velocity, none', readFloats(v, n * 32), g.floats('v_uncond0'));
for (const c of [...ctx, ...nctx]) c.dispose();
rope.dispose();
await flow.sample(x, coords, n, cond, uncond, 1374, { steps: g.ints('steps')[0] });
compare('sampled latent', readFloats(x, n * 32), g.floats('z'));
flow.dispose();
