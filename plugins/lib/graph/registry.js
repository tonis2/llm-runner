// The node types every loaded plugin has registered.
//
//   import { defineNodes } from '../lib/graph/registry.js';
//
//   defineNodes('flux', {
//     'flux.sample': {
//       title: 'Flux sampler', category: 'sampling',
//       inputs: { cond: 'CONDITIONING', model: 'MODEL', steps: 'INT=4', seed: 'INT=42' },
//       outputs: { latent: 'LATENT' },
//       async run(inputs, ctx) { ...; return { latent }; },
//     },
//   });
//
// `run` gets its inputs by name - wired values and settings alike - and returns
// its outputs by name. It owns nothing it was given: the executor disposes a
// value (anything with a `dispose()`) once nothing downstream needs it, so a
// node never disposes an input and does dispose its own scratch.
//
// `output: true` marks a sink (save, preview): what a graph runs for. A sink
// runs every time; everything else is cached by its settings and inputs.

import { parsePort } from './types.js';

const nodes = new Map();
const plugins = new Map();

export function defineNodes(plugin, defs) {
	for (const [type, def] of Object.entries(defs)) {
		if (typeof def.run !== 'function') throw new Error(`node ${type} has no run()`);
		const inputs = Object.entries(def.inputs ?? {}).map(([name, spec]) => ({ name, ...parsePort(spec, `${type}.${name}`) }));
		const outputs = Object.entries(def.outputs ?? {}).map(([name, spec]) => ({ name, ...parsePort(spec, `${type}.${name}`) }));
		nodes.set(type, {
			type,
			plugin,
			title: def.title ?? type,
			category: def.category ?? plugin,
			description: def.description ?? '',
			inputs,
			outputs,
			output: !!def.output,
			run: def.run,
		});
	}
	if (!plugins.has(plugin)) plugins.set(plugin, { name: plugin });
}

export function nodeType(type) {
	const def = nodes.get(type);
	if (!def) throw new Error(`no node type ${type}${nodes.size ? '' : ' (no plugins loaded)'}`);
	return def;
}

export function hasNodeType(type) { return nodes.has(type); }

// Every node type, without its run(): the catalogue the editor builds from.
export function catalogue() {
	const out = {};
	for (const [type, d] of nodes) {
		out[type] = {
			plugin: d.plugin, title: d.title, category: d.category, description: d.description,
			inputs: d.inputs, outputs: d.outputs, output: d.output,
		};
	}
	return out;
}
