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

// `profile: true` in the config runs every dispatch on its own and adds its
// wall time to a table by kernel (and, for the matmuls, by shape), which
// `profileReport` prints. It is slow; the numbers are each kernel alone.
const PROFILE = !!globalThis.__llm_config?.profile;
const profile = new Map();

// Record one dispatch. `workgroups` is [x] or [x, y] or [x, y, z].
export function dispatch(name, bindings, workgroups, push, independent = false, defines = null) {
	const wg = typeof workgroups === 'number' ? [workgroups, 1, 1] : workgroups;
	const k = kernel(name, defines);
	if (PROFILE) C.submit();
	const t0 = PROFILE ? llm.now() : 0;
	k.dispatch(bindings, {
		workgroups: [wg[0], wg[1] ?? 1, wg[2] ?? 1],
		push,
		independent,
	});
	dispatches++;
	if (PROFILE) {
		C.submit();
		let key = name;
		if (name.startsWith('matmul') && push) {
			const u = new Uint32Array(push.buffer, push.byteOffset, 3);
			key = `${name} ${u[2]}x${u[1]}->${u[0]}`;
		}
		const e = profile.get(key) ?? { ms: 0, n: 0 };
		e.ms += llm.now() - t0;
		e.n++;
		profile.set(key, e);
	}
}

export function profileReset() { profile.clear(); }

// The profile table, slowest first, as a share of the total.
export function profileReport(title = 'profile') {
	if (!PROFILE) return;
	const rows = [...profile].sort((a, b) => b[1].ms - a[1].ms);
	const total = rows.reduce((s, [, e]) => s + e.ms, 0);
	llm.print(`  [${title}] ${total.toFixed(0)}ms in kernels`);
	for (const [key, e] of rows) {
		llm.print(`    ${(100 * e.ms / total).toFixed(1).padStart(5)}%  ${e.ms.toFixed(1).padStart(9)}ms  ${String(e.n).padStart(6)}x  ${(e.ms / e.n).toFixed(3).padStart(8)}ms  ${key}`);
	}
}

// How many dispatches have been recorded, for a step's report.
export function dispatchCount() { return dispatches; }

// Run everything recorded and wait for it.
export function submit() { C.submit(); }

// A window's turn. Long work calls `await breathe()` between its blocks (a
// layer, a weight upload). Headless nothing is set and it does nothing (but
// submit, when `submitHere` says the caller wants its blocks cut there anyway);
// the studio sets `setBreather(() => three.nextFrame())`, and then what is
// recorded goes to the GPU and a frame is drawn while it runs. Work that is
// done at once, and a breath soon after the last, draws nothing: small blocks
// are not each held up by a frame.
const FRAME_MS = 40;
let breather = null;
let lastBreath = 0;
export function setBreather(fn) {
	breather = fn;
	lastBreath = llm.now();
}

// Weights uploaded one at a time with a breath after each: `spec` is
// `{ key: () => tensor }`, and the answer is `{ key: tensor }`.
export async function uploadEach(spec) {
	const out = {};
	for (const [key, make] of Object.entries(spec)) {
		out[key] = make();
		await breathe();
	}
	return out;
}

export async function breathe(submitHere = false) {
	if (!breather) {
		if (submitHere) C.submit();
		return;
	}
	C.submitAsync();
	if (C.busy() || llm.now() - lastBreath >= FRAME_MS) {
		await breather();
		C.submit();
		lastBreath = llm.now();
	}
}

// GPU-side copy of `bytes` bytes, in order with the dispatches.
export function copy(src, dst, bytes, srcOffset = 0, dstOffset = 0) {
	C.copy(src.buffer ?? src, dst.buffer ?? dst, { srcOffset, dstOffset, size: bytes });
}

export function zero(t) { C.zero(t.buffer ?? t); }
