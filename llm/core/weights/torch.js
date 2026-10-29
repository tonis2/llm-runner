// PyTorch checkpoints (`torch.save`'s zip, .pt / .ckpt / .pth), read without
// PyTorch and without running anything in them.
//
// The zip holds `<root>/data.pkl`, a pickle of whatever was saved, and one
// stored (uncompressed) entry `<root>/data/<key>` per storage. The pickle is read
// by an unpickler that builds plain values: dicts, lists, tuples, numbers,
// strings, and the tensors torch rebuilds through `_rebuild_tensor_v2`. Every
// other global (an OrderedDict, a Lightning AttributeDict, a TorchScript
// module's class, numpy scalars) becomes an inert `PickleObject` holding its
// arguments and state, and is never called.
//
//   import { openTorch } from './torch.js';
//   const m = openTorch('model.ckpt');        // an llm.js Model: has/info/shape/upload/floats
//   const w = m.upload('backbone.encoder.blocks.0.attn.qkv.weight', 'q8');
//
// The tensors are not copied into JavaScript: their offsets in the file go to
// the host as a safetensors-style table, and the mapped file is uploaded from
// as a safetensors file is. Only the zip's directory and the pickle are read.

import { Model } from '/core/llm.js';

const H = globalThis.__llm;

// QuickJS has no TextDecoder.
export function utf8(bytes) {
	let s = '';
	for (let i = 0; i < bytes.length;) {
		const c = bytes[i++];
		if (c < 0x80) s += String.fromCharCode(c);
		else if (c < 0xe0) s += String.fromCharCode(((c & 0x1f) << 6) | (bytes[i++] & 0x3f));
		else if (c < 0xf0) s += String.fromCharCode(((c & 0x0f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f));
		else {
			const cp = ((c & 0x07) << 18) | ((bytes[i++] & 0x3f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
			s += String.fromCodePoint(cp);
		}
	}
	return s;
}

// ------------------------------------------------------------------- zip

// Something to read a zip from: `size` and `read(offset, length)` -> Uint8Array.
export function bytesReader(bytes) {
	return { size: bytes.length, read: (offset, length) => bytes.subarray(offset, offset + length) };
}

export function fileReader(path) {
	const size = H.fileSize(path);
	if (size < 0) throw new Error(`${path} does not exist or cannot be read`);
	return { size, read: (offset, length) => H.readRange(path, offset, length) };
}

function view(bytes) { return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); }

// name -> { method, size, local, offset() }: `offset()` reads the entry's local
// header the first time it is asked, for where its data starts.
export function zipEntries(reader) {
	const tailLen = Math.min(reader.size, 65557);
	const tail = reader.read(reader.size - tailLen, tailLen);
	const tv = view(tail);
	let eocd = -1;
	for (let i = tail.length - 22; i >= 0; i--) {
		if (tv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
	}
	if (eocd < 0) throw new Error('not a zip file (no end of central directory)');
	let count = tv.getUint16(eocd + 10, true);
	let dirSize = tv.getUint32(eocd + 12, true);
	let dirAt = tv.getUint32(eocd + 16, true);
	// Zip64: the real numbers are in the zip64 end record.
	if (count === 0xffff || dirAt === 0xffffffff || dirSize === 0xffffffff) {
		const loc = eocd - 20;
		if (loc < 0 || tv.getUint32(loc, true) !== 0x07064b50) throw new Error('zip64 locator missing');
		const recAt = Number(tv.getBigUint64(loc + 8, true));
		const rv = view(reader.read(recAt, 56));
		if (rv.getUint32(0, true) !== 0x06064b50) throw new Error('bad zip64 end record');
		count = Number(rv.getBigUint64(32, true));
		dirSize = Number(rv.getBigUint64(40, true));
		dirAt = Number(rv.getBigUint64(48, true));
	}
	const dir = reader.read(dirAt, dirSize);
	const dv = view(dir);
	const entries = new Map();
	let p = 0;
	for (let n = 0; n < count; n++) {
		if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('bad zip central directory');
		const method = dv.getUint16(p + 10, true);
		let size = dv.getUint32(p + 24, true);
		const nameLen = dv.getUint16(p + 28, true);
		const extraLen = dv.getUint16(p + 30, true);
		const commentLen = dv.getUint16(p + 32, true);
		let local = dv.getUint32(p + 42, true);
		const name = utf8(dir.subarray(p + 46, p + 46 + nameLen));
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
		const entry = { method, size, local, at: -1 };
		entry.offset = () => {
			if (entry.at < 0) {
				const h = view(reader.read(local, 30));
				if (h.getUint32(0, true) !== 0x04034b50) throw new Error(`bad zip local header for ${name}`);
				entry.at = local + 30 + h.getUint16(26, true) + h.getUint16(28, true);
			}
			return entry.at;
		};
		entries.set(name, entry);
		p = end + commentLen;
	}
	return entries;
}

// -------------------------------------------------------------- unpickler

const MARK = Symbol('mark');

// A global the pickle constructs that is not a container or a tensor: its
// name, the arguments it was called with, and the state BUILD gave it.
export class PickleObject {
	constructor(name, args = null) {
		this.global = name;
		this.args = args;
		this.state = null;
	}
}

// What a storage type in a persistent id holds: safetensors' dtype name and the
// element size.
const STORAGES = {
	FloatStorage: ['F32', 4], DoubleStorage: ['F64', 8], HalfStorage: ['F16', 2], BFloat16Storage: ['BF16', 2],
	LongStorage: ['I64', 8], IntStorage: ['I32', 4], ShortStorage: ['I16', 2], CharStorage: ['I8', 1],
	ByteStorage: ['U8', 1], BoolStorage: ['BOOL', 1],
	UInt16Storage: ['U16', 2], UInt32Storage: ['U32', 4], UInt64Storage: ['U64', 8],
};

// A tensor as the pickle describes it: a view of a storage. `storage` is
// { key, dtype, size, numel }; offset, shape and stride count elements.
export class TorchTensor {
	constructor(storage, offset, shape, stride) {
		this.storage = storage;
		this.offset = offset;
		this.shape = shape;
		this.stride = stride;
	}
	get dtype() { return this.storage.dtype; }
	get elements() { return this.shape.reduce((a, b) => a * b, 1); }
	// Row-major with no gaps, the only layout the host reads.
	get contiguous() {
		let expect = 1;
		for (let i = this.shape.length - 1; i >= 0; i--) {
			if (this.shape[i] !== 1 && this.stride[i] !== expect) return false;
			expect *= this.shape[i];
		}
		return true;
	}
}

function callGlobal(name, args) {
	switch (name) {
		case 'collections OrderedDict': {
			const d = new Map();
			for (const [k, v] of args?.[0] ?? []) d.set(k, v);
			return d;
		}
		case 'torch._utils _rebuild_tensor_v2':
		case 'torch._utils _rebuild_tensor': {
			const [storage, offset, shape, stride] = args;
			return new TorchTensor(storage, offset, shape, stride);
		}
		// nn.Parameter around a tensor; the tensor is what matters.
		case 'torch._utils _rebuild_parameter':
		case 'torch._utils _rebuild_parameter_with_state':
			return args[0];
		// A tensor subclass: its rebuild function and that function's arguments.
		case 'torch._tensor _rebuild_from_type_v2': {
			const [fn, , fnArgs] = args;
			return fn?.global ? callGlobal(fn.global, fnArgs) : new PickleObject(name, args);
		}
		default:
			return new PickleObject(name, args);
	}
}

// The value a pickle holds. `persistent(pid)` answers a persistent id; torch's
// is ('storage', StorageType, key, location, numel).
export function unpickle(bytes, persistent = torchStorage) {
	const dv = view(bytes);
	const stack = [];
	const memo = new Map();
	let p = 0;
	const popMark = () => {
		const at = stack.lastIndexOf(MARK);
		if (at < 0) throw new Error('pickle: no mark on the stack');
		const items = stack.splice(at);
		items.shift();
		return items;
	};
	const top = () => stack[stack.length - 1];
	const text = (n) => { const s = utf8(bytes.subarray(p, p + n)); p += n; return s; };
	const line = () => { const e = bytes.indexOf(10, p); const s = utf8(bytes.subarray(p, e)); p = e + 1; return s; };
	const long = (n) => {
		let v = 0n;
		for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(bytes[p + i]);
		if (n > 0 && bytes[p + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
		p += n;
		return Number(v);
	};
	const setItems = (d, items) => {
		for (let i = 0; i < items.length; i += 2) {
			if (d instanceof Map) d.set(items[i], items[i + 1]);
			else if (d instanceof PickleObject) (d.items ??= new Map()).set(items[i], items[i + 1]);
			else throw new Error('pickle: SETITEM on something that is not a dict');
		}
	};
	const appendItems = (l, items) => {
		if (Array.isArray(l)) l.push(...items);
		else if (l instanceof PickleObject) (l.list ??= []).push(...items);
		else throw new Error('pickle: APPEND on something that is not a list');
	};
	for (;;) {
		if (p >= bytes.length) throw new Error('pickle ended without STOP');
		const op = bytes[p++];
		switch (op) {
			case 0x80: p++; break; // PROTO
			case 0x95: p += 8; break; // FRAME
			case 0x2e: return stack.pop(); // STOP
			case 0x7d: stack.push(new Map()); break; // EMPTY_DICT
			case 0x64: { const items = popMark(); const d = new Map(); setItems(d, items); stack.push(d); break; } // DICT
			case 0x5d: stack.push([]); break; // EMPTY_LIST
			case 0x6c: stack.push(popMark()); break; // LIST
			case 0x29: stack.push([]); break; // EMPTY_TUPLE
			case 0x8f: stack.push(new Set()); break; // EMPTY_SET
			case 0x90: { const items = popMark(); for (const v of items) top().add(v); break; } // ADDITEMS
			case 0x91: stack.push(new Set(popMark())); break; // FROZENSET
			case 0x28: stack.push(MARK); break; // MARK
			case 0x74: stack.push(popMark()); break; // TUPLE
			case 0x85: stack.push([stack.pop()]); break; // TUPLE1
			case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
			case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
			case 0x30: stack.pop(); break; // POP
			case 0x31: popMark(); break; // POP_MARK
			case 0x32: stack.push(top()); break; // DUP
			case 0x4e: stack.push(null); break; // NONE
			case 0x88: stack.push(true); break; // NEWTRUE
			case 0x89: stack.push(false); break; // NEWFALSE
			case 0x4b: stack.push(bytes[p]); p += 1; break; // BININT1
			case 0x4d: stack.push(dv.getUint16(p, true)); p += 2; break; // BININT2
			case 0x4a: stack.push(dv.getInt32(p, true)); p += 4; break; // BININT
			case 0x8a: { const n = bytes[p++]; stack.push(long(n)); break; } // LONG1
			case 0x8b: { const n = dv.getInt32(p, true); p += 4; stack.push(long(n)); break; } // LONG4
			case 0x49: { const s = line(); stack.push(s === '00' ? false : s === '01' ? true : Number(s)); break; } // INT
			case 0x4c: stack.push(Number(line().replace(/L$/, ''))); break; // LONG
			case 0x47: stack.push(dv.getFloat64(p, false)); p += 8; break; // BINFLOAT
			case 0x46: stack.push(Number(line())); break; // FLOAT
			case 0x58: { const n = dv.getUint32(p, true); p += 4; stack.push(text(n)); break; } // BINUNICODE
			case 0x8d: { const n = Number(dv.getBigUint64(p, true)); p += 8; stack.push(text(n)); break; } // BINUNICODE8
			case 0x8c: { const n = bytes[p++]; stack.push(text(n)); break; } // SHORT_BINUNICODE
			case 0x56: stack.push(line()); break; // UNICODE (raw-unicode-escape; ASCII in practice)
			case 0x54: { const n = dv.getInt32(p, true); p += 4; stack.push(text(n)); break; } // BINSTRING
			case 0x55: { const n = bytes[p++]; stack.push(text(n)); break; } // SHORT_BINSTRING
			case 0x53: stack.push(line().replace(/^'|'$/g, '')); break; // STRING
			case 0x42: { const n = dv.getUint32(p, true); p += 4; stack.push(bytes.slice(p, p + n)); p += n; break; } // BINBYTES
			case 0x43: { const n = bytes[p++]; stack.push(bytes.slice(p, p + n)); p += n; break; } // SHORT_BINBYTES
			case 0x8e: case 0x96: { const n = Number(dv.getBigUint64(p, true)); p += 8; stack.push(bytes.slice(p, p + n)); p += n; break; } // BINBYTES8, BYTEARRAY8
			case 0x71: memo.set(bytes[p++], top()); break; // BINPUT
			case 0x72: memo.set(dv.getUint32(p, true), top()); p += 4; break; // LONG_BINPUT
			case 0x70: memo.set(Number(line()), top()); break; // PUT
			case 0x94: memo.set(memo.size, top()); break; // MEMOIZE
			case 0x68: stack.push(memo.get(bytes[p++])); break; // BINGET
			case 0x6a: stack.push(memo.get(dv.getUint32(p, true))); p += 4; break; // LONG_BINGET
			case 0x67: stack.push(memo.get(Number(line()))); break; // GET
			case 0x63: { const mod = line(), name = line(); stack.push({ global: `${mod} ${name}` }); break; } // GLOBAL
			case 0x93: { const name = stack.pop(), mod = stack.pop(); stack.push({ global: `${mod} ${name}` }); break; } // STACK_GLOBAL
			case 0x52: { // REDUCE
				const args = stack.pop(), fn = stack.pop();
				if (!fn?.global) throw new Error('pickle: REDUCE of something that is not a global');
				stack.push(callGlobal(fn.global, args));
				break;
			}
			case 0x81: { // NEWOBJ
				const args = stack.pop(), cls = stack.pop();
				stack.push(cls?.global === 'collections OrderedDict' ? new Map() : new PickleObject(cls?.global ?? '?', args));
				break;
			}
			case 0x92: { // NEWOBJ_EX
				stack.pop();
				const args = stack.pop(), cls = stack.pop();
				stack.push(new PickleObject(cls?.global ?? '?', args));
				break;
			}
			case 0x51: stack.push(persistent(stack.pop())); break; // BINPERSID
			case 0x50: stack.push(persistent(line())); break; // PERSID
			case 0x73: { const v = stack.pop(), k = stack.pop(); setItems(top(), [k, v]); break; } // SETITEM
			case 0x75: { const items = popMark(); setItems(top(), items); break; } // SETITEMS
			case 0x61: { const v = stack.pop(); appendItems(top(), [v]); break; } // APPEND
			case 0x65: { const items = popMark(); appendItems(top(), items); break; } // APPENDS
			case 0x62: { // BUILD
				const state = stack.pop();
				const obj = top();
				// An OrderedDict's _metadata, a tensor's hooks: nothing a reader needs.
				if (obj instanceof PickleObject) obj.state = state;
				break;
			}
			default: throw new Error(`pickle opcode 0x${op.toString(16)} at ${p - 1} is not one this reader knows`);
		}
	}
}

export function torchStorage(pid) {
	if (!Array.isArray(pid) || pid[0] !== 'storage') throw new Error('pickle: a persistent id that is not a torch storage');
	const type = (pid[1]?.global ?? '').split(' ').pop();
	const known = STORAGES[type];
	if (!known) throw new Error(`a storage of type ${pid[1]?.global ?? '?'}, which this reader does not know`);
	return { key: String(pid[2]), dtype: known[0], size: known[1], numel: pid[4] };
}

// ------------------------------------------------------------- the file

// Every tensor in a value, by dotted path: dict keys, list indices, and the
// state of objects (a TorchScript module's attributes).
export function tensorsIn(value, prefix = '', out = new Map(), seen = new Set()) {
	if (value === null || typeof value !== 'object' || seen.has(value)) return out;
	seen.add(value);
	const join = (k) => (prefix ? `${prefix}.${k}` : String(k));
	if (value instanceof TorchTensor) out.set(prefix, value);
	else if (value instanceof Map) for (const [k, v] of value) tensorsIn(v, join(k), out, seen);
	else if (Array.isArray(value)) value.forEach((v, i) => tensorsIn(v, join(i), out, seen));
	else if (value instanceof PickleObject) {
		tensorsIn(value.state, prefix, out, seen);
		tensorsIn(value.items, prefix, out, seen);
		tensorsIn(value.list, prefix, out, seen);
	}
	return out;
}

// The archive's pickle, unpickled: { value, entries, root, reader }.
export function readTorch(path) {
	const reader = fileReader(path);
	const entries = zipEntries(reader);
	let pkl = null;
	for (const [name, e] of entries) {
		if (name.endsWith('/data.pkl') && (!pkl || name.length < pkl.name.length)) pkl = { name, e };
	}
	if (!pkl) throw new Error(`${path} is not a PyTorch checkpoint (no data.pkl in it)`);
	if (pkl.e.method !== 0) throw new Error(`${path}: its pickle is compressed, which this reader does not read`);
	const root = pkl.name.slice(0, -'data.pkl'.length);
	const value = unpickle(reader.read(pkl.e.offset(), pkl.e.size));
	return { value, entries, root, reader };
}

// A torch checkpoint as an llm.js Model. `select(value)` picks what holds the
// tensors; by default a top-level `state_dict` if there is one (Lightning,
// most training scripts), else the whole value. Tensors the host cannot read
// as they lie (not contiguous, or of an integer type) are left out of the
// table; `model.torch` keeps every one, `model.ints(name)` reads an
// integer tensor's values and `model.pickle` is the whole unpickled value.
export function openTorch(path, { select = null } = {}) {
	const { value, entries, root, reader } = readTorch(path);
	const holder = select ? select(value) : (value instanceof Map && value.has('state_dict') ? value.get('state_dict') : value);
	const all = tensorsIn(holder);
	if (all.size === 0) throw new Error(`${path} holds no tensors`);
	const where = (t) => {
		const e = entries.get(`${root}data/${t.storage.key}`);
		if (!e) throw new Error(`${path}: storage ${t.storage.key} is missing`);
		if (e.method !== 0) throw new Error(`${path}: storage ${t.storage.key} is compressed`);
		const start = e.offset() + t.offset * t.storage.size;
		return [start, start + t.elements * t.storage.size];
	};
	const header = {};
	const skipped = [];
	for (const [name, t] of all) {
		if (!t.contiguous || !['F32', 'F16', 'BF16'].includes(t.dtype) || t.shape.length > 4) { skipped.push(name); continue; }
		// A 0-d tensor is one element; the table wants a shape.
		header[name] = { dtype: t.dtype, shape: t.shape.length ? t.shape : [1], data_offsets: where(t) };
	}
	const model = new Model(path, H.modelOpen(path, JSON.stringify(header)));
	model.format = 'torch';
	model.torch = all;
	model.skipped = skipped;
	// The unpickled value itself, for what is not a tensor (names, settings).
	model.pickle = value;
	model.ints = (name) => {
		const t = all.get(name);
		if (!t) throw new Error(`${path} has no tensor named ${name}`);
		if (!t.contiguous) throw new Error(`${path}: ${name} is not contiguous`);
		const [start, end] = where(t);
		const raw = reader.read(start, end - start);
		const dv = view(raw), n = t.elements, out = new Array(n);
		for (let i = 0; i < n; i++) {
			switch (t.dtype) {
				case 'I64': out[i] = Number(dv.getBigInt64(i * 8, true)); break;
				case 'I32': out[i] = dv.getInt32(i * 4, true); break;
				case 'I16': out[i] = dv.getInt16(i * 2, true); break;
				case 'I8': out[i] = dv.getInt8(i); break;
				case 'U64': out[i] = Number(dv.getBigUint64(i * 8, true)); break;
				case 'U32': out[i] = dv.getUint32(i * 4, true); break;
				case 'U16': out[i] = dv.getUint16(i * 2, true); break;
				case 'U8': case 'BOOL': out[i] = raw[i]; break;
				default: throw new Error(`${name} is ${t.dtype}; read it with floats()`);
			}
		}
		return out;
	};
	// Elements `first` .. `first + count` of a float32 tensor, read from the
	// file and nothing else: rows of a matrix too large to want whole.
	model.floatRange = (name, first, count) => {
		const t = all.get(name);
		if (!t) throw new Error(`${path} has no tensor named ${name}`);
		if (t.dtype !== 'F32' || !t.contiguous) throw new Error(`${path}: ${name} is not contiguous float32`);
		if (first < 0 || first + count > t.elements) throw new Error(`${name}: elements ${first}..${first + count} of ${t.elements}`);
		const [start] = where(t);
		const raw = reader.read(start + first * 4, count * 4);
		return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + count * 4));
	};
	return model;
}
