// The studio's state, and the actions every panel shares. The widgets read it
// in render(); anything that changes it calls `app.changed()` (the graph) or
// `app.refresh()` (everything) so the widgets that show it draw again.

import { llm, image } from '/core/llm.js';
import { Executor, Cancelled } from '/core/graph/executor.js';
import { writeSettings } from '/core/graph/plugins.js';
import { decodeLatent, FORMATS, vaeKind, KIND_FORMAT } from '/core/vae/latents.js';
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

// A handler that reports instead of throwing: a throw inside a widget
// callback stops that callback for good.
export function safe(fn) {
	return (...args) => {
		try {
			const r = fn(...args);
			if (r && typeof r.catch === 'function') r.catch((e) => app.say(e.message ?? String(e), true));
		} catch (e) {
			app.say(e && e.message ? e.message : String(e), true);
		}
	};
}

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
async function previewFor(latent) {
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
	return image.fromTensor(await decodeLatent(path, latent), latent.w * f, latent.h * f, 3);
}

// Run the graph (or what `targets` need). Results stay cached between runs,
// so changing the seed re-samples without reloading the model.
export async function run(targets = null) {
	if (app.running) return;
	await runGraph(app.doc.graph(), { targets, show: true });
}

// Run `graph`, reporting as it goes, and say how it ended:
// { status: 'done' | 'cancelled' | 'error', error, node, outputs, results }.
// `label` goes before every status line; `show` puts the node states and
// pictures on the open graph's nodes, which only makes sense when `graph` is
// that graph.
export async function runGraph(graph, { targets = null, label = '', show = true } = {}) {
	if (app.running) return { status: 'error', error: 'already running', node: null, outputs: [] };
	const say = (text, error = false) => app.say(label ? `${label} · ${text}` : text, error);
	app.running = true;
	app.cancelRequested = false;
	app.nodeState = {};
	app.progress = null;
	say('running…');
	app.executor ??= new Executor({ keep: true, yieldFrame: () => three.nextFrame() });
	const t0 = llm.now();
	let current = null;
	try {
		const { results, ran, cached } = await app.executor.run(graph, {
			targets,
			cancelled: () => app.cancelRequested,
			async onProgress(e) {
				if (e.start) {
					current = e.node;
					if (show) app.nodeState[e.node] = 'running';
					app.progress = { node: e.node, step: 0, total: 0 };
					say(`running ${e.node}…`);
				} else if (e.done) {
					if (show) app.nodeState[e.node] = 'done';
					app.progress = null;
				} else {
					app.progress = { node: e.node, step: e.step, total: e.total };
					say(`${e.node}: step ${e.step}/${e.total}`);
					if (show && e.latent) {
						const img = await previewFor(e.latent());
						if (img) setImage(e.node, img);
					}
				}
				app.changed();
			},
		});
		if (show) {
			for (const id of cached) app.nodeState[id] = 'cached';
			for (const [id, r] of Object.entries(results)) {
				if (r && r.image && r.image.pixels) setImage(id, r.image);
				else if (r && typeof r.output === 'string' && llm.exists(r.output)) setImage(id, image.load(r.output));
			}
		}
		const outputs = Object.values(results).map((r) => r && r.output).filter((o) => typeof o === 'string');
		say(`done in ${((llm.now() - t0) / 1000).toFixed(1)}s: ${ran.length} ran, ${cached.length} cached${outputs.length ? ` · saved ${outputs.join(', ')}` : ''}`);
		return { status: 'done', error: null, node: null, outputs, results };
	} catch (e) {
		if (e instanceof Cancelled) {
			if (current) delete app.nodeState[current];
			say('cancelled');
			return { status: 'cancelled', error: null, node: current, outputs: [] };
		}
		const at = e && e.node ? e.node : current;
		const message = e && e.message ? e.message : String(e);
		if (at && show) {
			app.nodeState[at] = 'error';
			app.selected = at;
		}
		say(`${at ? at + ': ' : ''}${message}`, true);
		return { status: 'error', error: message, node: at, outputs: [] };
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
