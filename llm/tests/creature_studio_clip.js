// A studio clip animated as Rigsmith's Animate node does it, outside Rigsmith:
// found and fitted, and what the run read kept in `out`, to look at.
//
//   llm-runner tests/creature_studio_clip.js model=sam-3d-body-dinov3 rig=rig.glb \
//       finder=rig.finder.safetensors studio=reference/studio.json frames=clip.mp4 out=run/ \
//       [tracks_in=run/tracks.json] [draw=9] [request='{"flip":false}']
//
// `out` gets tracks.json (what the finder read), reads.png (`draw` frames with
// the fit and every read drawn on them), report.json and clip.glb. With
// `tracks_in` the joints are not found again, so it needs no GPU and no
// `model`: the frames are only drawn on, and laid on the reference picture.
// `finder` is still wanted then, for where its landmarks sit on the rig.
// `request` goes over the request Rigsmith sends (start on the reference, the
// mirrored read on).
import { llm } from '/core/llm.js';
import { utf8 } from '/core/graph/plugins.js';
import { animate } from '/plugins/creature/main.js';

const cfg = globalThis.__llm_config;
for (const k of ['rig', 'studio', 'frames', 'out']) if (!cfg[k]) throw new Error(`${k}= is needed (see the header)`);
if (!cfg.tracks_in && !(cfg.model && cfg.finder)) throw new Error('model= and finder= are needed to find the joints (or tracks_in=)');
const out = String(cfg.out).replace(/\/+$/, '');

const request = {
	rig: cfg.rig, studio: cfg.studio, frames: cfg.frames, fps: 24, name: 'clip', start: 'reference',
	draw: Number(cfg.draw ?? 9), outputs: ['glb', 'tracks'],
	...(cfg.finder ? { finder: cfg.finder } : {}),
	...(typeof cfg.request === 'string' ? JSON.parse(cfg.request) : cfg.request ?? {}),
};
if (cfg.tracks_in) request.views = JSON.parse(llm.readText(cfg.tracks_in)).map((tracks) => ({ tracks }));
const { reply, files } = await animate(request, { model: cfg.model });

for (const [name, bytes] of Object.entries(files)) llm.writeBytes(`${out}/${name === 'glb' ? 'clip.glb' : name}`, bytes);
llm.writeBytes(`${out}/report.json`, utf8(JSON.stringify({ name: reply.name, frames: reply.frames, fps: reply.fps, joints: reply.joints, report: reply.report })));

// The reads the fit left behind, the worst first.
const views = reply.report.studio.views.map((v) => v.name);
const off = reply.report.reads.flatMap((rows, v) => rows.map(([name, weight, miss, over]) => ({ name, view: views[v], weight, miss, over })))
	.filter((r) => r.weight >= 0.5 && r.over >= 0.25).sort((a, b) => b.over - a.over);
llm.print(`  ${off.length} sure reads the fit left behind${off.length ? ':' : ''}`);
for (const r of off.slice(0, 16)) llm.print(`    ${r.name} (${r.view}): ${(r.over * 100).toFixed(0)}% of frames over 30 px off, median ${r.miss} px, weight ${r.weight}`);
llm.print(`  kept in ${out}: ${[...Object.keys(files).map((n) => (n === 'glb' ? 'clip.glb' : n)), 'report.json'].join(', ')}`);
