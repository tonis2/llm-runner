// Hi3D's surface voxels of a mesh against the torch reference's (voxels.safetensors
// from ref_hi3d.py voxels, and small.safetensors' shrunken grid if it is there).
//
//   llm-runner tests/hi3d_voxels.js golden=build/hi3d/golden/example4 mesh=build/pixal3d/out/example4_1024.glb
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { readMeshGlb } from '/core/mesh/glb.js';
import { surfaceVoxels } from '/plugins/hi3d/voxels.js';

const cfg = globalThis.__llm_config;
const mesh = readMeshGlb(cfg.mesh);
llm.print(`mesh: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} faces`);

function check(name, golden, budget, grid) {
	const t0 = llm.now();
	const got = surfaceVoxels(mesh.positions, mesh.indices, { resolution: 128, budget });
	const time = llm.since(t0);
	const want = golden.ints('coords');
	const key = (c, i) => (c[3 * i + 2] * 2048 + c[3 * i + 1]) * 2048 + c[3 * i];
	const theirs = new Set();
	for (let i = 0; i < want.length / 3; i++) theirs.add(key(want, i));
	let shared = 0;
	for (let i = 0; i < got.coords.length / 3; i++) if (theirs.has(key(got.coords, i))) shared++;
	let same = got.coords.length === want.length;
	for (let i = 0; same && i < want.length; i++) same = got.coords[i] === want[i];
	llm.print(`${name}: grid ${got.grid} (reference ${grid}) after ${got.tries} tries, ours ${got.coords.length / 3}, reference ${want.length / 3}, shared ${shared}${same ? ', identical order' : ''}, in ${time}`);
	return got;
}

const vox = openSafetensors(`${cfg.golden}/voxels.safetensors`);
const got = check('128, budget 90000', vox, 90000, vox.ints('grid')[0]);
const c = vox.floats('center'), s = vox.floats('scale')[0];
llm.print(`frame: center ${Array.from(got.center).map((v) => v.toFixed(6))} (reference ${Array.from(c).map((v) => v.toFixed(6))}), scale ${got.scale} (reference ${s})`);
let small = null;
try { small = openSafetensors(`${cfg.golden}/small.safetensors`); } catch { /* not made yet */ }
if (small) check(`128, budget ${small.metadata.budget}`, small, Number(small.metadata.budget), Number(small.metadata.grid));
