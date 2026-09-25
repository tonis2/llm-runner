// llm-runner studio: build a pipeline out of nodes and run it.
//
//   llm-runner ui
//
// Every enabled plugin's nodes are in the list on the left; the canvas is the
// graph; the right side edits the selected node. Graphs are the same JSON
// `llm-runner run` takes, so a graph built here runs headless too.
//
// Keys: delete removes the selected node, ctrl+d duplicates it, ctrl+s saves,
// ctrl+r runs, f frames the graph, escape cancels a run.

import { llm } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { setBreather } from '../lib/gpu.js';
import { loadPlugins, readSettings, writeText } from '../lib/graph/plugins.js';
import { app, run, cancel, dropImages } from './app.js';
import { Doc, refreshTypes } from './doc.js';
import { GraphCanvas } from './canvas.js';
import { Palette, Inspector, StatusBar, Chrome, Dialogs, safe, removeSelected } from './panels.js';

try { three.window.title = 'llm-runner studio'; } catch { /* headless */ }

// Lay the interface out in the window's points rather than its pixels, so it
// is the same size on a high-density display.
function uiSize() {
	const r = three.renderSize();
	const scale = Math.max(1, three.window.scale || 1);
	if (Math.abs(three.ui.scale - scale) > 0.01) three.ui.scale = scale;
	return [Math.floor(r.width / scale), Math.floor(r.height / scale)];
}

app.actions = {
	newGraph() {
		dropImages();
		app.doc = new Doc();
		app.selected = null;
		app.nodeState = {};
		app.say('new graph');
		app.refresh();
	},

	open(path, { template = false } = {}) {
		const graph = JSON.parse(llm.readText(path));
		dropImages();
		app.doc = new Doc(graph);
		app.doc.path = template ? null : path;
		app.selected = null;
		app.nodeState = {};
		if (!template) {
			app.settings.last_graph = path;
			app.saveSettings();
		}
		app.widgets.canvas.autoFrame = true;
		app.widgets.canvas.frame();
		app.say(`${template ? 'template' : 'opened'} ${path}`);
		app.refresh();
	},

	save(path = app.doc.path) {
		if (!path) {
			const d = app.widgets.dialogs;
			d.saveAsText = 'graphs/graph.json';
			d.saveAs = true;
			return;
		}
		if (!path.endsWith('.json')) path += '.json';
		const slash = path.lastIndexOf('/');
		if (slash > 0) globalThis.__llm.makeDir(path.slice(0, slash));
		const text = JSON.stringify(app.doc.toJSON(), null, '\t') + '\n';
		writeText(path, text);
		app.doc.path = path;
		app.doc.dirty = false;
		app.settings.last_graph = path;
		app.saveSettings();
		app.say(`saved ${path}`);
		app.refresh();
	},
};

async function boot() {
	app.settings = readSettings();
	op.useMatrixCores(app.settings.matrix_cores ?? true);
	// Long GPU work hands the window a frame between its layers.
	setBreather(() => three.nextFrame());
	app.plugins = await loadPlugins(llm.root, app.settings);
	const types = refreshTypes();

	app.widgets.canvas = new GraphCanvas();
	app.widgets.palette = new Palette();
	app.widgets.inspector = new Inspector();
	app.widgets.status = new StatusBar();
	app.widgets.dialogs = new Dialogs();
	const chrome = new Chrome();
	app.widgets.chrome = chrome;
	chrome.resize(...uiSize());

	const config = globalThis.__llm_config ?? {};
	const first = config.graph ?? app.settings.last_graph ?? `${llm.root}/graph/templates/zimage-t2i.json`;
	try {
		if (first && llm.exists(first)) app.actions.open(first, { template: first.includes('/templates/') });
	} catch (e) {
		app.say(`could not open ${first}: ${e.message}`, true);
	}
	const loaded = app.plugins.filter((p) => p.loaded).map((p) => p.name);
	if (!app.statusError) app.say(`plugins: ${loaded.join(', ')} · ${Object.keys(types).length} node types`);

	chrome.mount();
	app.widgets.dialogs.mount();
	app.widgets.canvas.frame();

	three.systems.frame('studio.layout', () => chrome.resize(...uiSize()));

	const ctrl = () => three.input.isDown('ctrl') || three.input.isDown('control');
	three.onKeyDown('delete', safe(removeSelected));
	three.onKeyDown('escape', safe(cancel));
	three.onKeyDown('f', safe(() => { if (!ctrl()) app.widgets.canvas.frame(); }));
	three.onKeyDown('s', safe(() => { if (ctrl()) app.actions.save(); }));
	three.onKeyDown('r', safe(() => { if (ctrl()) run(); }));
	three.onKeyDown('d', safe(() => {
		if (!ctrl() || !app.selected) return;
		const c = app.doc.duplicate(app.selected);
		if (c) app.select(c.id);
		app.changed();
	}));

	// For checks without a person at the window: `shot=file.png` writes the
	// interface once it has drawn, and `run=true` runs the graph first.
	if (config.shot) {
		(async () => {
			for (let i = 0; i < 3; i++) await three.nextFrame();
			if (config.run) await run();
			if (config.select) app.select(config.select);
			for (let i = 0; i < 3; i++) await three.nextFrame();
			three.ui.flush();
			const s = three.screenshot(config.shot);
			llm.print(`studio: wrote ${s.path} · ${app.status}`);
			if (config.quit !== false) three.quit();
		})().catch((e) => { llm.print(`studio: ${e.stack ?? e}`); three.quit(); });
	}
}

boot().catch((e) => llm.print(`studio: could not start: ${e.stack ?? e.message ?? e}`));
