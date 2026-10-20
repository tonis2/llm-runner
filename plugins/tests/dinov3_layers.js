// DINOv3's `layers`: earlier blocks' patches laid beside the last one's. The
// last block's columns are the plain encoding's, exactly; each earlier block's
// are its own (not the last one's again) and through the final norm.
//
//   llm-runner tests/dinov3_layers.js model=sam-3d-body-dinov3
import { llm } from '../lib/llm.js';
import { loadBackbone } from '../creature/backbone.js';

const cfg = globalThis.__llm_config;
if (!cfg.model) throw new Error('model= is needed: the SAM 3D Body checkpoint (its DINOv3)');
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

const backbone = await loadBackbone(cfg.model);
const encoder = backbone.encoder;
const H = 96, W = 128, N = (H / 16) * (W / 16), D = encoder.dim, layers = [7, 20];
const pixels = llm.randomNormal(3 * H * W, 3);
const read = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);

const plain = await encoder.encode(pixels, H, W, { tokens: true, quiet: true });
const p = read(plain, N * D);
plain.dispose();
const wide = await encoder.encode(pixels, H, W, { tokens: true, quiet: true, layers });
const w = read(wide, N * D * 3);
wide.dispose();

let last = 0, same = 0, spread = 0;
for (let r = 0; r < N; r++) {
	for (let c = 0; c < D; c++) {
		last = Math.max(last, Math.abs(w[r * 3 * D + 2 * D + c] - p[r * D + c]));
		same = Math.max(same, Math.abs(w[r * 3 * D + c] - w[r * 3 * D + D + c]));
	}
	// Through the norm: each block's row has the norm's scale, not the residual's.
	let s = 0;
	for (let c = 0; c < D; c++) s += w[r * 3 * D + c] ** 2;
	spread = Math.max(spread, Math.sqrt(s / D));
}
check(last === 0, `the last block's columns are the plain encoding's (worst ${last.toExponential(2)})`);
check(same > 0.1, `block ${layers[0]}'s columns are its own, not block ${layers[1]}'s (they differ by ${same.toFixed(2)} at most)`);
check(Number.isFinite(spread) && spread < 50, `block ${layers[0]}'s rows went through the norm (RMS ${spread.toFixed(2)} at most)`);

backbone.dispose();
llm.print(failures ? `${failures} failed` : 'all passed');
