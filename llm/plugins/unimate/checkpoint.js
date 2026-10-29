// UniMate's released checkpoint, read as it is downloaded:
// huggingface.co/Linzhan/UniMate, <model>/checkpoints/checkpoint_step_<N>.pt.
//
// A .pt is PyTorch's zip: `<name>/data.pkl`, a pickle of the saved dict whose
// tensors point at `<name>/data/<key>`, each a raw little-endian storage. The
// entries are stored, not compressed, so a tensor is a view of the file's bytes.
// The zip and the pickle are read by `core/weights/torch.js`, whose unpickler builds
// plain values and never calls anything in the file.
//
// The weights used are the EMA ones, as the reference sampler does:
// `ema_state_dict.shadow_params` is a list in the order of the model's
// parameters, which is `model_state_dict`'s order with its repeats left out (a
// module shared by every block - the spectral RoPE - is saved once per block
// but is one parameter).

import { llm, f32 } from '/core/llm.js';
import { zipEntries, bytesReader, unpickle } from '/core/weights/torch.js';

// ------------------------------------------------------------- checkpoint

// The EMA weights of a UniMate .pt, by parameter name: `get(name)` answers
// { shape, data: Float32Array } (a view of the file), `upload(name)` a Tensor.
export class UniMateCheckpoint {
	constructor(path) {
		if (!llm.exists(path)) throw new Error(`${path} does not exist`);
		const t0 = llm.now();
		const bytes = llm.readBytes(path);
		const entries = zipEntries(bytesReader(bytes));
		let pkl = null;
		for (const [name, e] of entries) if (name.endsWith('/data.pkl')) pkl = { name, ...e };
		if (!pkl) throw new Error(`${path} is not a PyTorch checkpoint (no data.pkl)`);
		if (pkl.method !== 0) throw new Error(`${path}: compressed checkpoints are not read`);
		const root = pkl.name.slice(0, -'data.pkl'.length);
		const saved = unpickle(bytes.subarray(pkl.offset(), pkl.offset() + pkl.size));
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
			if (t.dtype !== 'F32') throw new Error(`${path}: a ${t.dtype} tensor; only float32 checkpoints are read`);
			const count = t.shape.reduce((a, b) => a * b, 1);
			// Contiguous, row-major: what every weight in this model is.
			let expect = 1;
			for (let i = t.shape.length - 1; i >= 0; i--) {
				if (t.shape[i] !== 1 && t.stride[i] !== expect) throw new Error(`${path}: a non-contiguous tensor`);
				expect *= t.shape[i];
			}
			const at = e.offset() + t.offset * 4;
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
