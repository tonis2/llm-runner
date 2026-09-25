// The studio's state, and the actions every panel shares. The widgets read it
// in render(); anything that changes it calls `app.changed()` (the graph) or
// `app.refresh()` (everything) so the widgets that show it draw again.

import { llm, image } from '../lib/llm.js';
import { Executor, Cancelled } from '../lib/graph/executor.js';
import { writeSettings } from '../lib/graph/plugins.js';
import { decodeLatent, FORMATS, vaeKind, KIND_FORMAT } from '../lib/latents.js';
import { Doc } from './doc.js';

export const app = {
	doc: new Doc(),
	selected: null,
	// Per node: 'running', 'done', 'cached' or 'error', for the last run.
	nodeState: {},
	// Per node: { tex, width, height } - its last picture (a sink's result, a
	// sampler's preview, a loaded image).
	images: {},
	progress: null,
	running: false,
	cancelRequested: false,
	status: 'ready',
	statusError: false,
	settings: {},
	plugins: [],
	executor: null,
	widgets: {},

	select(id) {
		if (this.selected === id) return;
		this.selected = id;
		this.refresh();
	},

	// The graph changed: redraw the canvas and the inspector.
	changed() {
		const w = this.widgets;
		if (w.canvas) w.canvas.version++;
		if (w.inspector) w.inspector.update();
		if (w.status) w.status.update();
	},

	refresh() {
		for (const w of Object.values(this.widgets)) w.update();
	},

	say(text, error = false) {
		this.status = String(text);
		this.statusError = error;
		if (error) llm.print(`studio: ${text}`);
		if (this.widgets.status) this.widgets.status.update();
	},

	saveSettings() {
		try {
			writeSettings(this.settings);
		} catch (e) {
			this.say(`could not save settings: ${e.message}`, true);
		}
	},
};

// Put a picture on a node: its texture is made once and replaced after that.
export function setImage(id, img) {
	const had = app.images[id];
	const tex = three.ui.texture(img, had ? had.tex : 0);
	app.images[id] = { tex, width: img.width, height: img.height };
	app.changed();
}

export function dropImages() {
	for (const img of Object.values(app.images)) three.ui.freeTexture(img.tex);
	app.images = {};
}

// ---------------------------------------------------------------------------
// Running

// A step preview needs a small decoder for the latent's format; taef1 reads
// Flux 1 latents (Z-Image's). Flux 2 has none yet, so its steps show progress
// only.
let previewDecoder = null;
function previewFor(latent) {
	const path = app.settings.preview_vae;
	if (!path) return null;
	if (!previewDecoder || previewDecoder.path !== path) {
		let format = null;
		try {
			const m = llm.open(path);
			format = KIND_FORMAT[vaeKind(m)];
			m.close();
		} catch { format = null; }
		previewDecoder = { path, format };
	}
	if (previewDecoder.format !== latent.format) return null;
	const f = FORMATS[latent.format].factor;
	return image.fromTensor(decodeLatent(path, latent), latent.w * f, latent.h * f, 3);
}

// Run the graph (or what `targets` need). Results stay cached between runs,
// so changing the seed re-samples without reloading the model.
export async function run(targets = null) {
	if (app.running) return;
	app.running = true;
	app.cancelRequested = false;
	app.nodeState = {};
	app.progress = null;
	app.say('running…');
	app.executor ??= new Executor({ keep: true, yieldFrame: () => three.nextFrame() });
	const t0 = llm.now();
	let current = null;
	try {
		const { results, ran, cached } = await app.executor.run(app.doc.graph(), {
			targets,
			cancelled: () => app.cancelRequested,
			onProgress(e) {
				if (e.start) {
					current = e.node;
					app.nodeState[e.node] = 'running';
					app.progress = { node: e.node, step: 0, total: 0 };
					app.say(`running ${e.node}…`);
				} else if (e.done) {
					app.nodeState[e.node] = 'done';
					app.progress = null;
				} else {
					app.progress = { node: e.node, step: e.step, total: e.total };
					app.say(`${e.node}: step ${e.step}/${e.total}`);
					if (e.latent) {
						const img = previewFor(e.latent());
						if (img) setImage(e.node, img);
					}
				}
				app.changed();
			},
		});
		for (const id of cached) app.nodeState[id] = 'cached';
		for (const [id, r] of Object.entries(results)) {
			if (r && r.image && r.image.pixels) setImage(id, r.image);
			else if (r && typeof r.output === 'string' && llm.exists(r.output)) setImage(id, image.load(r.output));
		}
		const saved = Object.values(results).map((r) => r && r.output).filter(Boolean);
		app.say(`done in ${((llm.now() - t0) / 1000).toFixed(1)}s: ${ran.length} ran, ${cached.length} cached${saved.length ? ` · saved ${saved.join(', ')}` : ''}`);
	} catch (e) {
		if (e instanceof Cancelled) {
			if (current) delete app.nodeState[current];
			app.say('cancelled');
		} else {
			const at = e && e.node ? e.node : current;
			if (at) {
				app.nodeState[at] = 'error';
				app.selected = at;
			}
			app.say(`${at ? at + ': ' : ''}${e && e.message ? e.message : e}`, true);
		}
	} finally {
		app.running = false;
		app.progress = null;
		app.refresh();
	}
}

export function cancel() {
	if (app.running) {
		app.cancelRequested = true;
		app.say('cancelling after this step…');
	}
}

// Let go of every model and result the runs have kept.
export function freeMemory() {
	if (app.running) return;
	if (app.executor) app.executor.clear();
	app.nodeState = {};
	app.say('freed the cached models and results');
	app.refresh();
}
