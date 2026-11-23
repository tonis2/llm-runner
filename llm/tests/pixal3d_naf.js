// NAF's upsampled DINO features against the torch reference's, on the 512
// picture: every 8th pixel of the 512 x 512 map, sampled at the pixel itself.
//
//   llm-runner tests/pixal3d_naf.js golden=build/pixal3d/golden/example4 weights="/path/pixal3d"
import { llm, f32 } from '/core/llm.js';
import { submit } from '/core/gpu.js';
import { Naf, sampleNaf, disposeEncoding } from '/plugins/pixal3d/naf.js';

const cfg = globalThis.__llm_config;
const gold = llm.open(`${cfg.golden}/naf512.safetensors`);
const prep = llm.open(`${cfg.golden}/prep.safetensors`);
const want = gold.floats('hr_sub8'); // [1024, 64, 64]
const lr = gold.floats('lr'); // [1024, 32, 32]
const C = 1024, G = 32, S = 512, SUB = 64;
const upload = (data) => {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
};
const rows = new Float32Array(G * G * C);
for (let c = 0; c < C; c++) for (let i = 0; i < G * G; i++) rows[i * C + c] = lr[c * G * G + i];
const points = new Float32Array(SUB * SUB * 2);
for (let i = 0; i < SUB; i++) for (let j = 0; j < SUB; j++) { points[(i * SUB + j) * 2] = 8 * j; points[(i * SUB + j) * 2 + 1] = 8 * i; }

const naf = new Naf(`${cfg.weights}/naf/naf_release.pth`);
naf.load();
const image = upload(prep.floats('image512'));
const t0 = llm.now();
const enc = naf.encode(image, S, G);
const values = upload(rows), at = upload(points), out = f32(SUB * SUB * C);
sampleNaf(enc, values, at, out, SUB * SUB);
submit();
const got = new Float32Array(out.buffer.readBytes(0, SUB * SUB * C * 4).buffer);
llm.print(`NAF encode + sample in ${llm.since(t0)}`);
let worst = 0, sum = 0, dot = 0, gg = 0, ww = 0, at_ = -1;
for (let p = 0; p < SUB * SUB; p++) {
	for (let c = 0; c < C; c++) {
		const a = got[p * C + c], b = want[c * SUB * SUB + p];
		const d = Math.abs(a - b);
		if (d > worst) { worst = d; at_ = p; }
		sum += d;
		dot += a * b;
		gg += a * a;
		ww += b * b;
	}
}
llm.print(`hr: worst ${worst.toFixed(4)} (at pixel ${8 * Math.floor(at_ / SUB)},${8 * (at_ % SUB)}), mean ${(sum / want.length).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}, |want| rms ${Math.sqrt(ww / want.length).toFixed(3)}`);
disposeEncoding(enc);
naf.dispose();
