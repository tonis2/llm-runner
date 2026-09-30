// Kimodo's motion checkpoint, from either of the two forms it comes in.
//
// A `.gguf` is kimodo.cpp's repack: the weights with `denoiser.backbone.`
// dropped from their names, the motion statistics as `stats.*` tensors and the
// hyperparameters as `kimodo.*` metadata.
//
// NVIDIA's release (huggingface.co/nvidia/Kimodo-*) is the same weights,
// byte for byte, as model.safetensors, with the motion statistics beside it:
//
//   model.safetensors                       the two stages, float32
//   stats/motion/{global_root,local_root,body}/{mean,std}.npy
//
// `model` may name the folder or its .safetensors, under any name. Nothing is
// converted: the tensors are read under their original names. The statistics
// are read from stats/motion/ beside the weights when it is there, and
// otherwise from stats.js, which carries them for the models it recognises,
// so for those the .safetensors alone is enough. The release's config.yaml is
// not read. The network's shape is read
// off the weights, the skeleton off the width of the motion they write, and
// what the weights cannot say (below) is how every Kimodo was trained. What a
// generation asks for (steps, length, seed) comes from the caller.

import { llm } from '../lib/llm.js';
import { KNOWN_STATS } from './stats.js';

// What the weights cannot say: the rate the motion was trained at, the length
// of the diffusion schedule, the attention heads, and how many of the prompt's
// tokens are kept.
const TRAINED = { fps: 30, baseSteps: 1000, heads: 8, textTokens: 50 };

const STAT_PARTS = [['global', 'global_root'], ['local', 'local_root'], ['body', 'body']];

// Upstream's skeletons by joint count, as the tables in skeletons.js name them.
// A motion frame is 9 root values and 12 per joint.
const SKELETONS_BY_JOINTS = { 30: 'soma30', 22: 'smplx22', 34: 'g1skel34' };

// What every caller reads: the open model, the prefix its tensor names carry,
// and the settings and statistics the denoiser needs.
//   { model, prefix, name, skeletonKey, motionDim, fps, baseSteps,
//     config: { width, heads, ffn, layers, textTokens }, stats }
export function openMotionCheckpoint(path) {
	return path.endsWith('.gguf') ? openGguf(path) : openRelease(path);
}

function openGguf(path) {
	const m = llm.open(path);
	if (m.meta('general.architecture') !== 'kimodo-motion') {
		m.close();
		throw new Error(`${path} is not a Kimodo motion GGUF`);
	}
	return {
		model: m,
		prefix: '',
		name: m.meta('general.name', 'Kimodo'),
		skeletonKey: m.meta('kimodo.skeleton'),
		motionDim: m.meta('kimodo.motion_dim'),
		fps: m.meta('kimodo.fps', 30),
		baseSteps: m.meta('kimodo.base_diffusion_steps', 1000),
		config: {
			width: m.meta('kimodo.hidden_size', 1024),
			heads: m.meta('kimodo.heads', 8),
			ffn: m.meta('kimodo.feed_forward_size', 2048),
			layers: m.meta('kimodo.layers', 16),
			textTokens: m.meta('kimodo.num_text_tokens', 50),
		},
		stats: {
			globalMean: m.floats('stats.global_root.mean'), globalStd: m.floats('stats.global_root.std'),
			localMean: m.floats('stats.local_root.mean'), localStd: m.floats('stats.local_root.std'),
			bodyMean: m.floats('stats.body.mean'), bodyStd: m.floats('stats.body.std'),
		},
	};
}

function openRelease(path) {
	const file = path.endsWith('.safetensors');
	const dir = file ? parentOf(path) : path.replace(/\/+$/, '');
	const weights = file ? path : `${dir}/model.safetensors`;
	if (!llm.exists(weights)) throw new Error(`${weights} does not exist`);

	const m = llm.open(weights);
	const prefix = 'denoiser.backbone.';
	const root = `${prefix}root_model.`;
	if (!m.has(`${root}input_linear.weight`)) {
		m.close();
		throw new Error(`${weights} is not a Kimodo motion model`);
	}
	// The motion's width is what the two stages write: the root stage the
	// global root, the body stage the rest.
	const motionDim = m.shape(`${root}output_linear.weight`)[1] + m.shape(`${prefix}body_model.output_linear.weight`)[1];
	const joints = (motionDim - 9) / 12;
	const skeletonKey = SKELETONS_BY_JOINTS[joints];
	if (!skeletonKey) {
		m.close();
		throw new Error(`${weights}: a ${motionDim}-wide motion (${joints} joints) is no skeleton Kimodo is known for`);
	}
	let stats;
	try {
		stats = releaseStats(dir, m, prefix, weights);
	} catch (e) {
		m.close();
		throw e;
	}
	let layers = 0;
	while (m.has(`${root}seqTransEncoder.layers.${layers}.linear1.weight`)) layers++;
	const embedding = m.shape(`${root}embed_text.weight`)[0];
	if (embedding !== 4096) {
		m.close();
		throw new Error(`${weights} expects ${embedding}-wide text embeddings, the encoder makes 4096`);
	}
	return {
		model: m,
		prefix,
		name: (file ? path.slice(dir.length + 1).replace(/\.safetensors$/, '') : dir.slice(dir.lastIndexOf('/') + 1)) || 'Kimodo',
		skeletonKey,
		motionDim,
		fps: TRAINED.fps,
		baseSteps: TRAINED.baseSteps,
		config: {
			width: m.shape(`${root}input_linear.weight`)[1],
			heads: TRAINED.heads,
			ffn: m.shape(`${root}seqTransEncoder.layers.0.linear1.weight`)[1],
			layers,
			textTokens: TRAINED.textTokens,
		},
		stats,
	};
}

// stats/motion/ beside the weights, or the statistics stats.js carries for
// these weights.
function releaseStats(dir, m, prefix, weights) {
	const beside = (part, which) => `${dir}/stats/motion/${part}/${which}.npy`;
	if (STAT_PARTS.some(([, part]) => llm.exists(beside(part, 'mean')))) {
		const stats = {};
		for (const [key, part] of STAT_PARTS) {
			for (const which of ['mean', 'std']) {
				const npy = beside(part, which);
				if (!llm.exists(npy)) throw new Error(`${npy} is missing: stats/motion/ beside the weights is incomplete`);
				stats[key + (which === 'mean' ? 'Mean' : 'Std')] = readNpy(npy);
			}
		}
		return stats;
	}
	const fingerprint = weightsFingerprint(m, prefix);
	const known = KNOWN_STATS.find((k) => k.fingerprint === fingerprint);
	if (!known) {
		throw new Error(`${weights} is not a Kimodo model this plugin carries the motion statistics for `
			+ `(fingerprint ${fingerprint}): put the stats/motion/ folder from its Hugging Face page beside it`);
	}
	llm.print(`  [kimodo motion] statistics of ${known.name}, carried by the plugin`);
	const stats = {};
	for (const [key] of STAT_PARTS) {
		stats[`${key}Mean`] = Float32Array.from(known[`${key}Mean`]);
		stats[`${key}Std`] = Float32Array.from(known[`${key}Std`]);
	}
	return stats;
}

// FNV-1a (32-bit) over the float32 bytes of the two stages' output biases:
// small, read in a moment, and different in every trained model.
function weightsFingerprint(m, prefix) {
	let h = 0x811c9dc5;
	for (const stage of ['root_model', 'body_model']) {
		const values = m.floats(`${prefix}${stage}.output_linear.bias`);
		const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
		for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193) >>> 0;
	}
	return h.toString(16).padStart(8, '0');
}

function parentOf(path) {
	const slash = path.lastIndexOf('/');
	return slash < 0 ? '.' : path.slice(0, slash);
}

// A one-dimensional little-endian float .npy (NumPy's format, versions 1-3)
// as float32.
function readNpy(path) {
	const bytes = llm.readBytes(path);
	const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
	if (bytes.length < 10 || magic.some((b, i) => bytes[i] !== b)) throw new Error(`${path} is not a .npy file`);
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const major = bytes[6];
	const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
	const start = (major === 1 ? 10 : 12) + headerLength;
	let header = '';
	for (let i = start - headerLength; i < start; i++) header += String.fromCharCode(bytes[i]);

	const descr = /'descr'\s*:\s*'([^']*)'/.exec(header)?.[1];
	if (/'fortran_order'\s*:\s*True/.test(header)) throw new Error(`${path}: Fortran-ordered arrays are not supported`);
	const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number) ?? [];
	const count = shape.reduce((a, b) => a * b, 1);
	const size = descr === '<f8' ? 8 : descr === '<f4' ? 4 : 0;
	if (!size) throw new Error(`${path}: element type ${descr} is not supported (float32 or float64 expected)`);
	if (start + count * size > bytes.length) throw new Error(`${path} is shorter than its header says`);

	const out = new Float32Array(count);
	for (let i = 0; i < count; i++) {
		out[i] = size === 8 ? view.getFloat64(start + i * 8, true) : view.getFloat32(start + i * 4, true);
	}
	return out;
}
