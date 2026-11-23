// Where one V-Flow pass spends its time (run with profile=true for the table).
//
//   llm-runner tests/lato2_bench.js golden=build/lato2/golden/example0 weights="/path/lato2" profile=true [linear=q8]
import { llm, f32 } from '/core/llm.js';
import { submit, profileReport, profileReset } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { VertexFlow } from '/plugins/lato2/vflow.js';
import { upload, ropeTables } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config;
const dino = llm.open(`${cfg.golden}/dino.safetensors`);
const g = openSafetensors(`${cfg.golden}/vflow.safetensors`);
const coords = Int32Array.from(g.ints('coords')), n = coords.length / 3;
const flow = new VertexFlow(`${cfg.weights}/vflow.pt`, { linear: cfg.linear ?? 'f32' });
await flow.load();
const ctx = flow.context(upload(dino.floats('cond')), 1374);
const rope = ropeTables(coords, n, 64);
const x = upload(g.floats('noise')), v = f32(n * 32);
flow.forward(x, n, rope, 1, 200, ctx, v);
submit();
profileReset();
const t0 = llm.now();
flow.forward(x, n, rope, 1, 200, ctx, v);
submit();
llm.print(`${n} points: one pass ${llm.since(t0)}`);
profileReport('V-Flow pass');
