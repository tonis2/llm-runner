// Pixal3D's conditioning against the torch reference's: the camera, the
// sparse-structure stage's global tokens and proj features (16^3 grid on the
// 512 picture), and - once shape512.safetensors exists - the shape stage's
// [n, 2048] proj features at the sparse structure's voxels.
//
//   llm-runner tests/pixal3d_condition.js golden=build/pixal3d/golden/example4 image=build/pixal3d/inputs/example4.png weights="/path/pixal3d"
import { llm, image } from '/core/llm.js';
import { preprocess } from '/plugins/pixal3d/image.js';
import { Conditioner, camera } from '/plugins/pixal3d/condition.js';
import { openSafetensors } from '/core/weights/torch.js';

const cfg = globalThis.__llm_config;
const read = (t) => new Float32Array(t.buffer.readBytes(0, t.elements * 4).buffer);
function compare(name, got, want) {
	let worst = 0, sum = 0, dot = 0, gg = 0, ww = 0;
	for (let i = 0; i < want.length; i++) {
		const d = Math.abs(got[i] - want[i]);
		worst = Math.max(worst, d);
		sum += d;
		dot += got[i] * want[i];
		gg += got[i] * got[i];
		ww += want[i] * want[i];
	}
	llm.print(`${name}: ${got.length === want.length ? '' : `LENGTH ${got.length} vs ${want.length}, `}worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}`);
}

const cam = camera(0.2);
llm.print(`camera: distance ${cam.distance} (reference 4.983322211629619)`);
const pre = preprocess(image.load(cfg.image));
const cond = new Conditioner(cfg.weights);
await cond.load();
const prep = llm.open(`${cfg.golden}/prep.safetensors`);
const pic = await cond.picture(pre, 512);
const g = cond.global(pic);
compare('ss global', read(g), prep.floats('ss_global'));
const p = cond.proj(pic, 16, cam);
compare('ss proj', read(p), prep.floats('ss_proj'));
g.dispose();
p.dispose();
cond.disposePicture(pic);

{
	let sh = null;
	try { sh = openSafetensors(`${cfg.golden}/shape512.safetensors`); } catch { llm.print('no shape512 golden yet'); }
	if (sh) {
		const coords = Int32Array.from(sh.ints('coords'));
		const xyz = new Int32Array((coords.length / 4) * 3);
		for (let t = 0; t < coords.length / 4; t++) for (let a = 0; a < 3; a++) xyz[3 * t + a] = coords[4 * t + 1 + a];
		const pic2 = await cond.picture(pre, 512, { naf: 512 });
		const p2 = cond.proj(pic2, 32, cam, xyz);
		const got = read(p2), want = sh.floats('proj'), n = want.length / 2048;
		const half = (a, h) => { const o = new Float32Array(n * 1024); for (let t = 0; t < n; t++) o.set(a.subarray(t * 2048 + h * 1024, t * 2048 + h * 1024 + 1024), t * 1024); return o; };
		compare('shape512 proj, DINO half', half(got, 0), half(want, 0));
		compare('shape512 proj, NAF half', half(got, 1), half(want, 1));
		p2.dispose();
		cond.disposePicture(pic2);
	}
}
cond.dispose();
