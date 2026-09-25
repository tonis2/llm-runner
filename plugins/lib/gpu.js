// Kernels by name, and the one call every op goes through.
//
// A kernel is `plugins/lib/kernels/<name>.shady`, compiled on first use (and
// cached on disk by the engine after that). `common.shady` — the f16 and Q8_0
// decoders — is put in front of every source, so a kernel calls them as if it
// had written them.
//
//   dispatch('silu', [x], [groups(n, 256)], pc('u', n));
//
// Push blocks are packed by type letter — `u` a uint, `i` an int, `f` a float —
// because a JavaScript number says nothing about which of the three the shader
// declared.

import { llm } from './llm.js';

const C = three.compute;
const sources = new Map();
const kernels = new Map();
let common = null;

function source(name) {
	let text = sources.get(name);
	if (text === undefined) {
		text = llm.readText(llm.path(`lib/kernels/${name}.shady`));
		sources.set(name, text);
	}
	return text;
}

// The sizes of flash attention for a head width (a multiple of 16, <= 128).
export function flashDefines(hd) {
	return { HD: hd, HD1: hd + 1, KV: 32 * (hd + 1), QT: 16 * hd, QPT: (16 * hd) / 256, KPT: (32 * hd) / 256 };
}

// What a templated kernel is compiled with when no values are given.
const DEFAULT_DEFINES = { flash_attention: flashDefines(128) };

// The compiled kernel for `name`, compiled once per run and per set of
// `defines`: a kernel whose sizes are a parameter spells them `$NAME` in its
// source, and each distinct set of values is its own pipeline.
//
//   kernel('flash_attention', { HD: 64, ... })
export function kernel(name, defines = null) {
	defines = defines ?? DEFAULT_DEFINES[name] ?? null;
	const key = defines ? `${name}:${JSON.stringify(defines)}` : name;
	let k = kernels.get(key);
	if (k) return k;
	if (common === null) common = source('common');
	let text = source(name);
	if (defines) {
		text = text.replace(/\$([A-Z][A-Z0-9_]*)/g, (whole, id) => (id in defines ? String(defines[id]) : whole));
	}
	k = C.kernel(common + '\n' + text, { name: defines ? key : name });
	kernels.set(key, k);
	return k;
}

// Compile a list of kernels now, so a failure in one is reported before a
// model has spent a minute loading.
export function precompile(names) {
	for (const name of names) kernel(name);
}

// Pack a push block: `pc('uuf', 1, 2, 0.5)`.
const scratch = new DataView(new ArrayBuffer(256));
export function pc(types, ...values) {
	if (types.length !== values.length) throw new Error(`pc('${types}') was given ${values.length} values`);
	for (let i = 0; i < types.length; i++) {
		const v = values[i];
		switch (types[i]) {
			case 'u': scratch.setUint32(i * 4, v >>> 0, true); break;
			case 'i': scratch.setInt32(i * 4, v | 0, true); break;
			case 'f': scratch.setFloat32(i * 4, v, true); break;
			default: throw new Error(`pc: unknown type letter '${types[i]}'`);
		}
	}
	return new Uint8Array(scratch.buffer.slice(0, types.length * 4));
}

// Workgroups to cover `n` items at `per` items a group.
export function groups(n, per) { return Math.ceil(n / per); }

let dispatches = 0;

// Record one dispatch. `workgroups` is [x] or [x, y] or [x, y, z].
export function dispatch(name, bindings, workgroups, push, independent = false, defines = null) {
	const wg = typeof workgroups === 'number' ? [workgroups, 1, 1] : workgroups;
	kernel(name, defines).dispatch(bindings, {
		workgroups: [wg[0], wg[1] ?? 1, wg[2] ?? 1],
		push,
		independent,
	});
	dispatches++;
}

// How many dispatches have been recorded, for a step's report.
export function dispatchCount() { return dispatches; }

// Run everything recorded and wait for it.
export function submit() { C.submit(); }

// GPU-side copy of `bytes` bytes, in order with the dispatches.
export function copy(src, dst, bytes, srcOffset = 0, dstOffset = 0) {
	C.copy(src.buffer ?? src, dst.buffer ?? dst, { srcOffset, dstOffset, size: bytes });
}

export function zero(t) { C.zero(t.buffer ?? t); }
