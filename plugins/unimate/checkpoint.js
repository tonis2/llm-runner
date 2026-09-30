// UniMate's released checkpoint, read as it is downloaded:
// huggingface.co/Linzhan/UniMate, <model>/checkpoints/checkpoint_step_<N>.pt.
//
// A .pt is PyTorch's zip: `<name>/data.pkl`, a pickle of the saved dict whose
// tensors point at `<name>/data/<key>`, each a raw little-endian storage. The
// entries are stored, not compressed, so a tensor is a view of the file's bytes.
// The pickle is read by a small unpickler that knows the three globals torch
// writes into a state dict (OrderedDict, FloatStorage, _rebuild_tensor_v2);
// anything else is refused, so nothing in the file is ever run.
//
// The weights used are the EMA ones, as the reference sampler does:
// `ema_state_dict.shadow_params` is a list in the order of the model's
// parameters, which is `model_state_dict`'s order with its repeats left out (a
// module shared by every block - the spectral RoPE - is saved once per block
// but is one parameter).

import { llm, f32 } from '../lib/llm.js';
import { utf8 } from './rig.js';

// ------------------------------------------------------------------- zip

const decoder = { decode: utf8 };

function zipEntries(bytes) {
	const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	let eocd = -1;
	for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
		if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
	}
	if (eocd < 0) throw new Error('not a zip file (no end of central directory)');
	let count = dv.getUint16(eocd + 10, true);
	let dirAt = dv.getUint32(eocd + 16, true);
	// Zip64: the real numbers are in the zip64 end record.
	if (count === 0xffff || dirAt === 0xffffffff) {
		const loc = eocd - 20;
		if (dv.getUint32(loc, true) !== 0x07064b50) throw new Error('zip64 locator missing');
		const rec = Number(dv.getBigUint64(loc + 8, true));
		count = Number(dv.getBigUint64(rec + 32, true));
		dirAt = Number(dv.getBigUint64(rec + 48, true));
	}
	const entries = new Map();
	let p = dirAt;
	for (let n = 0; n < count; n++) {
		if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('bad zip central directory');
		const method = dv.getUint16(p + 10, true);
		let size = dv.getUint32(p + 24, true);
		const nameLen = dv.getUint16(p + 28, true);
		const extraLen = dv.getUint16(p + 30, true);
		const commentLen = dv.getUint16(p + 32, true);
		let local = dv.getUint32(p + 42, true);
		const name = decoder.decode(bytes.subarray(p + 46, p + 46 + nameLen));
		// Zip64 extra field: 64-bit sizes and offset, in that order, for the
		// 32-bit fields that are all ones.
		let e = p + 46 + nameLen;
		const end = e + extraLen;
		while (e + 4 <= end) {
			const id = dv.getUint16(e, true), len = dv.getUint16(e + 2, true);
			if (id === 0x0001) {
				let q = e + 4;
				if (size === 0xffffffff) { q += 8; size = Number(dv.getBigUint64(q, true)); q += 8; }
				else if (dv.getUint32(p + 20, true) === 0xffffffff) q += 8;
				if (local === 0xffffffff) local = Number(dv.getBigUint64(q, true));
			}
			e += 4 + len;
		}
		entries.set(name, { method, size, local });
		p = end + commentLen;
	}
	// Where each entry's data starts, from its local header.
	for (const entry of entries.values()) {
		const h = entry.local;
		if (dv.getUint32(h, true) !== 0x04034b50) throw new Error('bad zip local header');
		entry.offset = h + 30 + dv.getUint16(h + 26, true) + dv.getUint16(h + 28, true);
	}
	return entries;
}

// -------------------------------------------------------------- unpickler

const MARK = Symbol('mark');

class OrderedDict extends Map {}

// What the three allowed globals build.
function callGlobal(name, args) {
	switch (name) {
		case 'collections OrderedDict': return new OrderedDict();
		case 'torch._utils _rebuild_tensor_v2': {
			const [storage, offset, shape, stride] = args;
			return { storage, offset, shape, stride };
		}
		default: throw new Error(`the checkpoint's pickle calls ${name}, which is not a tensor`);
	}
}

function unpickle(bytes) {
	const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const stack = [];
	const memo = new Map();
	let p = 0;
	const popMark = () => {
		const at = stack.lastIndexOf(MARK);
		const items = stack.splice(at);
		items.shift();
		return items;
	};
	for (;;) {
		const op = bytes[p++];
		switch (op) {
			case 0x80: p++; break; // PROTO
			case 0x2e: return stack.pop(); // STOP
			case 0x7d: stack.push(new Map()); break; // EMPTY_DICT
			case 0x5d: stack.push([]); break; // EMPTY_LIST
			case 0x29: stack.push([]); break; // EMPTY_TUPLE
			case 0x28: stack.push(MARK); break; // MARK
			case 0x74: stack.push(popMark()); break; // TUPLE
			case 0x85: stack.push([stack.pop()]); break; // TUPLE1
			case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
			case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
			case 0x4e: stack.push(null); break; // NONE
			case 0x88: stack.push(true); break; // NEWTRUE
			case 0x89: stack.push(false); break; // NEWFALSE
			case 0x4b: stack.push(bytes[p]); p += 1; break; // BININT1
			case 0x4d: stack.push(dv.getUint16(p, true)); p += 2; break; // BININT2
			case 0x4a: stack.push(dv.getInt32(p, true)); p += 4; break; // BININT
			case 0x8a: { // LONG1
				const n = bytes[p++];
				let v = 0n;
				for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(bytes[p + i]);
				if (n > 0 && bytes[p + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
				p += n;
				stack.push(Number(v));
				break;
			}
			case 0x47: stack.push(dv.getFloat64(p, false)); p += 8; break; // BINFLOAT
			case 0x58: { const n = dv.getUint32(p, true); p += 4; stack.push(decoder.decode(bytes.subarray(p, p + n))); p += n; break; } // BINUNICODE
			case 0x8c: { const n = bytes[p++]; stack.push(decoder.decode(bytes.subarray(p, p + n))); p += n; break; } // SHORT_BINUNICODE
			case 0x71: memo.set(bytes[p++], stack[stack.length - 1]); break; // BINPUT
			case 0x72: memo.set(dv.getUint32(p, true), stack[stack.length - 1]); p += 4; break; // LONG_BINPUT
			case 0x94: memo.set(memo.size, stack[stack.length - 1]); break; // MEMOIZE
			case 0x68: stack.push(memo.get(bytes[p++])); break; // BINGET
			case 0x6a: stack.push(memo.get(dv.getUint32(p, true))); p += 4; break; // LONG_BINGET
			case 0x63: { // GLOBAL: "module\nname\n"
				let e = bytes.indexOf(10, p);
				const mod = decoder.decode(bytes.subarray(p, e));
				const e2 = bytes.indexOf(10, e + 1);
				const name = decoder.decode(bytes.subarray(e + 1, e2));
				p = e2 + 1;
				stack.push({ global: `${mod} ${name}` });
				break;
			}
			case 0x93: { const name = stack.pop(), mod = stack.pop(); stack.push({ global: `${mod} ${name}` }); break; } // STACK_GLOBAL
			case 0x52: { // REDUCE
				const args = stack.pop(), fn = stack.pop();
				if (!fn?.global) throw new Error('pickle REDUCE of something that is not a global');
				stack.push(callGlobal(fn.global, args));
				break;
			}
			case 0x51: { // BINPERSID: ('storage', FloatStorage, key, location, numel)
				const pid = stack.pop();
				if (!Array.isArray(pid) || pid[0] !== 'storage') throw new Error('unknown persistent id in the checkpoint');
				const type = pid[1]?.global ?? '';
				if (type !== 'torch FloatStorage') throw new Error(`storage of ${type}: only float32 checkpoints are read`);
				stack.push({ key: pid[2], numel: pid[4] });
				break;
			}
			case 0x73: { const v = stack.pop(), k = stack.pop(); stack[stack.length - 1].set(k, v); break; } // SETITEM
			case 0x75: { // SETITEMS
				const items = popMark();
				const d = stack[stack.length - 1];
				for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]);
				break;
			}
			case 0x61: { const v = stack.pop(); stack[stack.length - 1].push(v); break; } // APPEND
			case 0x65: { const items = popMark(); stack[stack.length - 1].push(...items); break; } // APPENDS
			case 0x62: stack.pop(); break; // BUILD: state for an object (OrderedDict's _metadata); dropped
			default: throw new Error(`the checkpoint's pickle uses opcode 0x${op.toString(16)}, which this reader does not know`);
		}
	}
}

// ------------------------------------------------------------- checkpoint

// The EMA weights of a UniMate .pt, by parameter name: `get(name)` answers
// { shape, data: Float32Array } (a view of the file), `upload(name)` a Tensor.
export class UniMateCheckpoint {
	constructor(path) {
		if (!llm.exists(path)) throw new Error(`${path} does not exist`);
		const t0 = llm.now();
		const bytes = llm.readBytes(path);
		const entries = zipEntries(bytes);
		let pkl = null;
		for (const [name, e] of entries) if (name.endsWith('/data.pkl')) pkl = { name, ...e };
		if (!pkl) throw new Error(`${path} is not a PyTorch checkpoint (no data.pkl)`);
		if (pkl.method !== 0) throw new Error(`${path}: compressed checkpoints are not read`);
		const root = pkl.name.slice(0, -'data.pkl'.length);
		const saved = unpickle(bytes.subarray(pkl.offset, pkl.offset + pkl.size));
		const state = saved.get('model_state_dict');
		const ema = saved.get('ema_state_dict');
		if (!state) throw new Error(`${path} has no model_state_dict: not a UniMate training checkpoint`);

		// Parameter names in order: the state dict's, each storage once.
		const seen = new Set(), names = [];
		for (const [name, t] of state) {
			if (seen.has(t.storage.key)) continue;
			seen.add(t.storage.key);
			names.push(name);
		}
		const shadow = ema?.get('shadow_params');
		if (shadow && shadow.length !== names.length) {
			throw new Error(`${path}: ${shadow.length} EMA weights for ${names.length} parameters`);
		}
		this.tensors = new Map();
		const view = (t) => {
			const e = entries.get(`${root}data/${t.storage.key}`);
			if (!e || e.method !== 0) throw new Error(`${path}: storage ${t.storage.key} missing or compressed`);
			const count = t.shape.reduce((a, b) => a * b, 1);
			// Contiguous, row-major: what every weight in this model is.
			let expect = 1;
			for (let i = t.shape.length - 1; i >= 0; i--) {
				if (t.shape[i] !== 1 && t.stride[i] !== expect) throw new Error(`${path}: a non-contiguous tensor`);
				expect *= t.shape[i];
			}
			const at = e.offset + t.offset * 4;
			if (at % 4) throw new Error(`${path}: tensor data not aligned`);
			return { shape: t.shape, data: new Float32Array(bytes.buffer, bytes.byteOffset + at, count) };
		};
		names.forEach((name, i) => this.tensors.set(name, view(shadow ? shadow[i] : state.get(name))));
		this.ema = !!shadow;
		this.step = saved.get('step') ?? null;
		this.bytes = bytes;
		llm.print(`  [unimate] checkpoint step ${this.step ?? '?'} (${this.ema ? 'EMA' : 'raw'} weights, ${names.length} tensors) read in ${llm.since(t0)}`);
	}

	has(name) { return this.tensors.has(name); }

	get(name) {
		const t = this.tensors.get(name);
		if (!t) throw new Error(`the UniMate checkpoint has no ${name}`);
		return t;
	}

	shape(name) { return this.get(name).shape; }

	// A copy of the floats, for work done on the host.
	floats(name) { return Float32Array.from(this.get(name).data); }

	upload(name) {
		const t = this.get(name);
		const g = f32(t.data.length, t.shape.slice().reverse(), name);
		g.buffer.write(t.data);
		return g;
	}

	// The file's bytes are only needed until the weights are on the GPU.
	release() {
		this.bytes = null;
		this.tensors = new Map([...this.tensors].map(([k, t]) => [k, { shape: t.shape, data: null }]));
	}
}
