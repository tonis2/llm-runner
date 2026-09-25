// Finding plugins and loading their nodes.
//
// A plugin is a directory under the plugins root with a `plugin.json`:
//
//   { "name": "flux", "version": "1.0", "description": "...",
//     "nodes": "nodes.js", "main": "main.js" }
//
// `nodes` is the module that registers its node types (and must not call
// llm.plugin); `main` is the optional CLI/server plugin. User settings live in
// ~/.config/llm-runner/settings.json:
//
//   { "disabled_plugins": ["depth"], "output_dir": "/home/me/Pictures/llm", "model_dirs": [...] }

import { llm } from '../llm.js';

const H = globalThis.__llm;

export function settingsPath() {
	const home = H.homeDir();
	return home ? `${home}/.config/llm-runner/settings.json` : '';
}

export function readSettings() {
	const path = settingsPath();
	if (!path || !llm.exists(path)) return {};
	try {
		return JSON.parse(llm.readText(path));
	} catch (e) {
		llm.print(`llm-runner: ${path} is not JSON (${e.message}); using the defaults`);
		return {};
	}
}

export function writeSettings(settings) {
	const path = settingsPath();
	if (!path) throw new Error('no home directory to keep settings in');
	H.makeDir(path.slice(0, path.lastIndexOf('/')));
	llm.writeBytes(path, utf8(JSON.stringify(settings, null, '\t') + '\n'));
}

function utf8(text) {
	const bytes = [];
	for (const ch of text) {
		const c = ch.codePointAt(0);
		if (c < 0x80) bytes.push(c);
		else if (c < 0x800) bytes.push(0xc0 | (c >> 6), 0x80 | (c & 63));
		else if (c < 0x10000) bytes.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
		else bytes.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
	}
	return new Uint8Array(bytes);
}

// Every plugin directory's manifest: [{ name, dir, version, description, nodes, main, enabled }].
export function listPlugins(root = llm.root, settings = readSettings()) {
	const disabled = new Set(settings.disabled_plugins ?? []);
	const out = [];
	for (const entry of H.listDir(root)) {
		if (!entry.dir) continue;
		const manifest = `${root}/${entry.name}/plugin.json`;
		if (!llm.exists(manifest)) continue;
		let m;
		try {
			m = JSON.parse(llm.readText(manifest));
		} catch (e) {
			out.push({ name: entry.name, dir: entry.name, error: `plugin.json: ${e.message}`, enabled: false });
			continue;
		}
		const name = m.name ?? entry.name;
		out.push({ ...m, name, dir: entry.name, enabled: !disabled.has(name) });
	}
	return out;
}

// Import the nodes of every enabled plugin. A plugin that fails to load is
// reported and skipped; the rest still load.
export async function loadPlugins(root = llm.root, settings = readSettings()) {
	const plugins = listPlugins(root, settings);
	for (const p of plugins) {
		if (!p.enabled || !p.nodes || p.error) continue;
		try {
			await import(`../../${p.dir}/${p.nodes}`);
			p.loaded = true;
		} catch (e) {
			p.error = e.message ?? String(e);
			llm.print(`llm-runner: plugin ${p.name} did not load: ${p.error}`);
		}
	}
	return plugins;
}
