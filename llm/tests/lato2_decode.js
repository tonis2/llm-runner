// LATO.2 after V-Flow against the torch reference, each stage fed the
// reference's own input: the V-VAE decoder (expansion, vertex voxels, each
// level's survivors), the offsets, the voxel encoder, T-Flow, the T-VAE's
// features and edges, and the faces.
//
//   llm-runner tests/lato2_decode.js golden=build/lato2/golden/example0 weights="/path/lato2"
import { llm, f32 } from '/core/llm.js';
import { submit } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { VertexDecoder } from '/plugins/lato2/vvae.js';
import { VoxelEncoder, TopologyFlow, TopologyDecoder } from '/plugins/lato2/topology.js';
import { edgesToFaces } from '/plugins/lato2/faces.js';
import { upload, readFloats } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config, W = cfg.weights;
const only = cfg.only ? String(cfg.only).split(',') : null;
const run = (name) => !only || only.includes(name);
function compare(name, got, want) {
	let worst = 0, sum = 0, mag = 0;
	for (let i = 0; i < want.length; i++) { const d = Math.abs(got[i] - want[i]); worst = Math.max(worst, d); sum += d; mag += Math.abs(want[i]); }
	llm.print(`${name}: worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)} against mean |x| ${(mag / want.length).toFixed(3)}`);
}
// How many of `got`'s rows of three are among `want`'s.
function sameRows(name, got, want) {
	const set = new Set();
	for (let i = 0; i < want.length; i += 3) set.add(`${want[i]},${want[i + 1]},${want[i + 2]}`);
	let hit = 0;
	for (let i = 0; i < got.length; i += 3) if (set.has(`${got[i]},${got[i + 1]},${got[i + 2]}`)) hit++;
	llm.print(`${name}: ${got.length / 3} ours, ${want.length / 3} reference, ${hit} shared`);
}

if (run('vvae')) {
	const g = openSafetensors(`${cfg.golden}/vvae.safetensors`);
	const coords = Int32Array.from(g.ints('coords')), n = coords.length / 3;
	const dec = new VertexDecoder(`${W}/vvae.pt`, `${W}/offset_head.pt`);
	dec.load();
	const record = {};
	const r = await dec.decode(upload(g.floats('latent')), coords, n, { record });
	compare('expanded latent', record.expanded, g.floats('expanded'));
	compare('vertex-voxel logits', record.vtxLogits, g.floats('vtx_logits'));
	const mask = g.ints('vtx_mask');
	let agree = 0;
	mask.forEach((m, i) => { if ((record.vtxLogits[i] >= 0) === (m === 1)) agree++; });
	llm.print(`vertex voxels: ${agree} of ${n} agree (${mask.filter((m) => m === 1).length} kept by the reference)`);
	for (let i = 0; i < 4; i++) {
		const want = Int32Array.from(g.ints(`l${i}_coords`));
		sameRows(`level ${i} points`, record[`l${i}`].coords, want);
		if (record[`l${i}`].coords.length === want.length && record[`l${i}`].coords.every((v, j) => v === want[j])) {
			compare(`level ${i} features`, record[`l${i}`].feats, g.floats(`l${i}_feats`));
		}
	}
	if (r.coords.length === g.ints('l3_coords').length) compare('offsets', r.offsets, g.floats('offsets'));
	dec.dispose();
}

if (run('voxel')) {
	const g = openSafetensors(`${cfg.golden}/voxel_encoder.safetensors`);
	const coords = Int32Array.from(g.ints('coords'));
	const enc = new VoxelEncoder(`${W}/voxel_encoder.pt`);
	enc.load();
	const { tokens, count } = enc.run(coords, coords.length / 3);
	compare(`voxel tokens (${count})`, readFloats(tokens, count * enc.dim), g.floats('tokens'));
	enc.dispose();
}

if (run('tflow')) {
	const g = openSafetensors(`${cfg.golden}/tflow.safetensors`);
	const verts = Int32Array.from(g.ints('verts')), n = verts.length / 3;
	const flow = new TopologyFlow(`${W}/tflow.pt`);
	await flow.load();
	const cond = upload(g.floats('cond'));
	const p = flow.prepare(verts, n, cond, 512);
	const x = upload(g.floats('noise')), v = f32(n * flow.zDim);
	flow.forward(x, n, 0, p, v);
	compare('T-Flow velocity at t 0', readFloats(v, n * flow.zDim), g.floats('v0'));
	p.dispose();
	await flow.sample(x, verts, n, cond, 512, { steps: g.ints('steps')[0] });
	compare('T-Flow latent (over the scale)', readFloats(x, n * flow.zDim), g.floats('z'));
	flow.dispose();
}

if (run('tvae')) {
	const g = openSafetensors(`${cfg.golden}/tvae.safetensors`);
	const verts = Int32Array.from(g.ints('verts')), n = verts.length / 3;
	const dec = new TopologyDecoder(`${W}/tvae.pt`);
	dec.load();
	const feats = dec.features(upload(g.floats('z')), verts, n);
	compare('T-VAE features', readFloats(feats, n * dec.C), g.floats('feats'));
	const edges = dec.edges(feats, n, 0);
	const want = g.ints('edges'), wset = new Set();
	for (let e = 0; e < want.length; e += 2) wset.add(want[e] * n + want[e + 1]);
	let hit = 0;
	for (let e = 0; e < edges.length; e += 2) if (wset.has(edges[e] * n + edges[e + 1])) hit++;
	llm.print(`edges: ${edges.length / 2} ours, ${want.length / 2} reference, ${hit} shared`);
	// faces from the reference's own edges, so only the face code is compared
	const refEdges = Int32Array.from(want);
	const sorted = (f) => { const o = []; for (let i = 0; i < f.length; i += 3) o.push(...[f[i], f[i + 1], f[i + 2]].sort((a, b) => a - b)); return o; };
	sameRows('faces, no fill', sorted(edgesToFaces(refEdges, n, verts, { fill: false })), sorted(g.ints('faces_plain')));
	const t0 = llm.now();
	const filled = edgesToFaces(refEdges, n, verts, { fill: true });
	llm.print(`  fill in ${llm.since(t0)}`);
	sameRows('faces, filled', sorted(filled), sorted(g.ints('faces')));
	dec.dispose();
}
