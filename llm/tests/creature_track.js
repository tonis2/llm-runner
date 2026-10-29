// The creature tracker against a track made before (tests/_dino_track.js's
// output, or another run's): the same frames and labels, each point's
// distance from the reference frame by frame, and the time a frame.
//
//   llm-runner tests/creature_track.js model=sam-3d-body-dinov3 labels=labels.json ref=track.json \
//       [frames=dir] [region=x1,y1,x2,y2] [width=1280] [scale=1] [out=tracks.json]
//
// `labels` is a request's ({ frames, labels, region, width }) or the old
// script's ({ joints, points, images }: points on frame 0). `scale` maps this
// run's frame pixels onto the reference's: ref = (p - region corner) * scale
// when the reference was tracked on crops of `region`.
import { llm } from '/core/llm.js';
import { writeText } from '/core/graph/plugins.js';
import { loadBackbone } from '/plugins/creature/backbone.js';
import { Tracker } from '/plugins/creature/track.js';
import { frameSource, readLabels } from '/plugins/creature/request.js';

const cfg = globalThis.__llm_config;
const list = (v) => (v === undefined || v === null || v === '' ? null : Array.isArray(v) ? v.map(Number) : String(v).split(',').map(Number));
const doc = JSON.parse(llm.readText(cfg.labels));
const request = doc.labels ? doc : { frames: doc.images, labels: { 0: Object.fromEntries(doc.joints.map((j, i) => [j, doc.points[i]])) } };
const frames = frameSource(cfg.frames ?? request.frames);
const { labels } = readLabels(request);
const region = list(cfg.region) ?? request.region ?? null;
const width = Number(cfg.width ?? request.width ?? 1280);
const ref = cfg.ref ? JSON.parse(llm.readText(cfg.ref)) : null;
const scale = Number(cfg.scale ?? 1);

const backbone = await loadBackbone(cfg.model);
const tracker = new Tracker(backbone.encoder);
// Only as far as the reference goes.
const count = ref ? Math.min(frames.count, Math.max(...ref.frames) + 1) : frames.count;
const tracks = await tracker.track({ ...frames, count, image: (k) => frames.image(k) }, { labels, region, width });
if (cfg.out) writeText(cfg.out, JSON.stringify(tracks));
tracker.dispose();
backbone.dispose();

let failures = 0;
if (ref) {
	const h = tracks.hypotheses.find((x) => x.direction > 0 && x.from === ref.frames[0]) ?? tracks.hypotheses[0];
	const o = region ? [region[0], region[1]] : [0, 0];
	const errs = [], perPoint = new Map();
	ref.frames.forEach((f, row) => {
		const i = h.frames.indexOf(f);
		if (i < 0) return;
		ref.joints.forEach((j, p) => {
			const q = h.tracks[i][tracks.points.indexOf(j)];
			if (!q) return;
			const e = Math.hypot((q[0] - o[0]) * scale - ref.tracks[row][p][0], (q[1] - o[1]) * scale - ref.tracks[row][p][1]);
			errs.push(e);
			perPoint.set(j, Math.max(perPoint.get(j) ?? 0, e));
		});
	});
	errs.sort((a, b) => a - b);
	const median = errs[Math.floor(errs.length / 2)], p90 = errs[Math.floor(errs.length * 0.9)], max = errs[errs.length - 1];
	llm.print(`  against ${cfg.ref}: ${errs.length} points, median ${median.toFixed(2)}px, 90% ${p90.toFixed(2)}px, max ${max.toFixed(1)}px (reference pixels)`);
	llm.print(`  worst points: ${[...perPoint].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([j, e]) => `${j} ${e.toFixed(1)}`).join(', ')}`);
	const want = Number(cfg.max_median ?? 1);
	if (!(median <= want)) {
		llm.print(`FAIL: median ${median.toFixed(2)}px over ${want}px`);
		failures++;
	}
}
llm.print(failures ? `${failures} failed` : 'ok');
