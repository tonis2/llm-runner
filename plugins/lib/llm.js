// llm-runner — the host API a pipeline plugin is written against.
//
// `three.compute` is the GPU: buffers, kernels, dispatches. This is everything
// around it that a model needs: files, the weights inside them, a tokenizer,
// seeded noise and images. The host verbs are `globalThis.__llm` (see
// runner/host.c3); what is here is the part that reads well.
//
//   import { llm, GGML } from '../lib/llm.js';
//
//   const model = llm.open(config.model);
//   const w = model.upload('img_in.weight');          // a Tensor on the GPU
//   const tok = llm.open(config.text_model).tokenizer();
//
//   llm.plugin({ name: 'flux', load(config) { ... }, generate(config) { ... } });

const H = globalThis.__llm;
const C = three.compute;

// GGML tensor types, as GGUF numbers them.
export const GGML = {
	F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
	Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, BF16: 30,
};
export const GGML_NAME = Object.fromEntries(Object.entries(GGML).map(([k, v]) => [v, k]));

// A tensor on the GPU: the buffer a kernel binds, plus what it holds.
//
// `type` is a GGML number for whatever the bytes are *now* — an upload `as:
// 'f32'` of a Q8_0 weight is F32 — and `shape` is GGUF order, fastest-varying
// dimension first, whichever file it came from.
export class Tensor {
	constructor(buffer, type, shape, name = '') {
		this.buffer = buffer;
		this.type = type;
		this.shape = shape;
		this.name = name;
	}
	get bytes() { return this.buffer.byteLength; }
	get elements() { return this.shape.reduce((a, b) => a * b, 1); }
	get typeName() { return GGML_NAME[this.type] ?? String(this.type); }
	view(byteOffset, byteSize) { return this.buffer.view(byteOffset, byteSize); }
	get _binding() { return this.buffer._binding; }
	dispose() { this.buffer.dispose(); }
}

// A float32 scratch tensor of `count` elements.
export function f32(count, shape = [count], name = '') {
	return new Tensor(C.f32(count), GGML.F32, shape, name);
}

// Weight types `ops.matmul` has a kernel for, uploaded as they are.
export const NATIVE_TYPES = new Set([GGML.F32, GGML.Q8_0]);

const SAFETENSORS_TYPE = { F32: GGML.F32, F16: GGML.F16, BF16: GGML.BF16 };

// A model file, mapped: its metadata and its tensor table. The bytes stay in the
// mapping until something is uploaded.
export class Model {
	constructor(path, info) {
		this.path = path;
		this.handle = info.handle;
		this.format = info.format;
		this.metadata = info.metadata;
		this.tensors = new Map();
		for (const t of info.tensors) this.tensors.set(t.name, t);
	}

	has(name) { return this.tensors.has(name); }

	// The tensor's description, or a thrown error naming what is missing.
	info(name) {
		const t = this.tensors.get(name);
		if (!t) throw new Error(`${this.path} has no tensor named ${name}`);
		return t;
	}

	// GGUF-order shape, whatever the file's own order is.
	shape(name) {
		const t = this.info(name);
		return this.format === 'gguf' ? t.shape.slice() : t.shape.slice().reverse();
	}

	// The file's own type of a tensor, as a GGML number (safetensors F8 and the
	// like come back as their dtype string).
	type(name) {
		const t = this.info(name);
		return this.format === 'gguf' ? t.type : (SAFETENSORS_TYPE[t.type] ?? t.type);
	}

	meta(key, fallback) {
		const v = this.metadata[key];
		return v === undefined ? fallback : v;
	}

	// A metadata array in full: strings, a Float32Array or a Uint32Array.
	array(key) { return H.modelArray(this.handle, key); }

	// Put a tensor on the GPU.
	//
	//   as: 'raw'  — the bytes as they are in the file (quantised stays quantised)
	//       'f32'  — dequantised / widened to float
	//       'conv' — f32, and a [out, in, kh, kw] conv weight laid out [kh, kw, in, out]
	//       'q8'   — f32, re-quantised to Q8_0
	//       'auto' — 'raw' for a type the matmul kernels read natively
	//                (NATIVE_TYPES), 'f32' for anything else
	upload(name, as = 'raw') {
		if (as === 'auto') as = NATIVE_TYPES.has(this.type(name)) ? 'raw' : 'f32';
		const shape = this.shape(name);
		const r = H.tensorUpload(this.handle, name, as);
		let type;
		if (as === 'raw') type = this.type(name);
		else if (as === 'q8') type = GGML.Q8_0;
		else type = GGML.F32;
		let tshape = shape;
		if (as === 'conv' && shape.length === 4) tshape = shape; // [kw, kh, in, out] fastest-first
		return new Tensor(C.adopt(r.buffer, 'bytes', r.bytes), type, tshape, name);
	}

	// A small tensor on the host, as a Float32Array.
	floats(name) { return H.tensorFloats(this.handle, name); }

	// Rows of an embedding table, looked up on the host and written into `into`
	// (a Tensor or ComputeBuffer) from its start as [ids.length, dim] floats.
	embedRows(name, ids, into) {
		const u32 = ids instanceof Uint32Array ? ids : Uint32Array.from(ids);
		const buffer = into instanceof Tensor ? into.buffer : into;
		H.embedRows(this.handle, name, new Uint8Array(u32.buffer, u32.byteOffset, u32.byteLength), buffer._h);
	}

	tokenizer() { return new Tokenizer(H.tokenizerOpen(this.handle)); }

	close() { H.modelClose(this.handle); }
}

export class Tokenizer {
	constructor(info) {
		this.handle = info.handle;
		this.vocab = info.vocab;
		this.bos = info.bos;
		this.eos = info.eos;
		this.pad = info.pad;
	}
	// Token ids as a Uint32Array. `specials` reads `<|im_start|>`-style markers
	// as the single tokens they are.
	encode(text, specials = true) { return H.tokenize(this.handle, text, specials); }
	// The id of one special token's text, or 0 when there is none.
	find(text) { return H.tokenFind(this.handle, text); }
	decode(ids) {
		const u32 = ids instanceof Uint32Array ? ids : Uint32Array.from(ids);
		return H.detokenize(this.handle, new Uint8Array(u32.buffer, u32.byteOffset, u32.byteLength));
	}
}

function asBytes(floats) {
	return new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
}

export const image = {
	// { width, height, channels, pixels: Uint8Array }
	load(path) { return H.imageLoad(path); },
	decode(bytes) { return H.imageDecode(bytes); },
	savePng(path, img) { H.imageSavePng(path, img); },
	encodePng(img) { return H.imageEncodePng(img); },
	// Resize the short side to cover w x h and centre-crop to exactly that.
	cropTo(img, w, h) { return H.imageResize(img, 'crop', w, h); },
	// Long side to `longSide`, both sides rounded down to a multiple of 16.
	fit16(img, longSide) { return H.imageResize(img, 'fit16', longSide, 0); },
	// To exactly w x h, aspect not kept.
	resize(img, w, h) { return H.imageResize(img, 'exact', w, h); },
	// [3, h, w] floats, in [-1, 1] or, with range 'unit', [0, 1].
	toTensor(img, range = 'signed') { return H.imageToTensor(img, range); },
	// [c, h, w] floats in [0, 1] to an 8-bit image.
	fromTensor(floats, width, height, channels = 3) {
		return H.tensorToImage(asBytes(floats), width, height, channels);
	},
};

export const llm = {
	// The plugins directory, as the runner was pointed at it.
	root: globalThis.__llm_root ?? 'plugins',

	// A line on stdout, now — console.log is collected and shown when the run ends.
	print(...parts) { H.print(parts.map((p) => (typeof p === 'string' ? p : JSON.stringify(p))).join(' ')); },
	// Milliseconds since the runner started.
	now() { return H.now(); },

	readText(path) { return H.readText(path); },
	readBytes(path) { return H.readBytes(path); },
	writeBytes(path, bytes) { H.writeBytes(path, bytes instanceof Uint8Array ? bytes : asBytes(bytes)); },
	exists(path) { return H.exists(path); },
	base64Encode(bytes) { return H.base64Encode(bytes); },
	base64Decode(text) { return H.base64Decode(text); },

	// A path under the plugins root.
	path(relative) { return `${this.root}/${relative}`; },

	open(path) { return new Model(path, H.modelOpen(path)); },

	// Gaussian noise for a seed — the same numbers the C3 pipelines drew.
	randomNormal(count, seed) { return H.randomNormal(count, seed); },

	image,

	// Register the plugin this file is. The runner calls `load(config)` once and
	// then `generate(config)` — or, as a server, `handle(request)` per request
	// (or `generate` with the request's JSON over the config, when there is no
	// `handle`).
	plugin(definition) { globalThis.__llm_plugin = definition; },

	// Server mode: the request being answered, and the answer.
	request() { return H.request(); },
	respond(status, contentType, body) { H.respond(status, contentType, body); },

	// Seconds, to one decimal, since `start` (an llm.now()).
	since(start) { return ((H.now() - start) / 1000).toFixed(2) + 's'; },
};
