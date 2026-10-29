// SAM 3D Body's weights, as they are downloaded: huggingface.co/facebook/sam-3d-body-dinov3
// (gated) is `model.ckpt`, a Lightning checkpoint whose `state_dict` holds
// every tensor, with the MHR rig beside it in `assets/mhr_model.pt`. A
// safetensors conversion of the same state dict (the same names, no prefix)
// opens too.

import { llm } from '/core/llm.js';
import { openTorch } from '/core/weights/torch.js';

// `path` is the downloaded folder, its model.ckpt, or a .safetensors file.
export function openWeights(path) {
	let file = path;
	if (globalThis.__llm.listDir(path).length) {
		const names = globalThis.__llm.listDir(path).map((e) => e.name);
		const pick = names.includes('model.ckpt') ? 'model.ckpt' : names.find((n) => n.endsWith('.safetensors'));
		if (!pick) throw new Error(`${path} has neither model.ckpt nor a .safetensors file`);
		file = `${path}/${pick}`;
	}
	if (!llm.exists(file)) throw new Error(`${file} does not exist`);
	const t0 = llm.now();
	const m = file.endsWith('.safetensors') ? llm.open(file) : openTorch(file);
	if (!m.has('backbone.encoder.cls_token')) {
		m.close();
		throw new Error(`${file} is not SAM 3D Body's checkpoint (no backbone.encoder.cls_token)`);
	}
	llm.print(`  SAM 3D Body: ${m.tensors.size} tensors in ${file}, read in ${llm.since(t0)}`);
	return m;
}

// Beside the checkpoint, as the release lays it out.
export function mhrPath(path) {
	const dir = path.endsWith('.ckpt') || path.endsWith('.safetensors') ? path.slice(0, path.lastIndexOf('/')) : path;
	return `${dir}/assets/mhr_model.pt`;
}
