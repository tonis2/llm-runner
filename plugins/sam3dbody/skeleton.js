// MHR as a motion skeleton: the joints with fixed bone lengths and rest
// rotations of identity, the way glTF clips, BVH and Kimodo's clean-up want a
// skeleton, and each frame's model parameters as parent-local turns on it.
//
// SAM 3D Body predicts the body's proportions anew every frame (model
// parameters 130 on: bone lengths, widths, scales), so a clip holds them to
// their median over its frames; what is left frame to frame is joint angles.
//
// MHR's rest pose (all parameters zero) stands y up, faces +z and has the
// character's left at +x - SAM 3D Body's camera frame (y down, z forward)
// turned half a turn about x. The rest pose here is that pose: each joint's
// offset from its parent is in that frame, and a frame's turn of joint j is
// its MHR world rotation times the inverse of its rest one.
//
// Joint 0 ('body_world') never moves and is left out: 'root', the pelvis, is
// the root.

import { mul, conjugate, unit } from '../kimodo/rotation.js';
import { JOINTS, MODEL_PARAMS } from './mhr.js';

// Model parameters from here on are the body's proportions.
export const PROPORTIONS = 130;
const FIRST = 1;

// The median of each proportion over `paramsList` (model parameters [204] a frame).
export function clipProportions(paramsList) {
	const out = new Float64Array(MODEL_PARAMS - PROPORTIONS);
	for (let i = 0; i < out.length; i++) {
		const v = paramsList.map((p) => p[PROPORTIONS + i]).sort((a, b) => a - b);
		const n = v.length;
		out[i] = n === 0 ? 0 : n % 2 ? v[(n - 1) >> 1] : (v[n / 2 - 1] + v[n / 2]) / 2;
	}
	return out;
}

export class MotionSkeleton {
	// `proportions` [74], clipProportions' answer (zeros: MHR's mean body).
	constructor(mhr, proportions = new Float64Array(MODEL_PARAMS - PROPORTIONS)) {
		this.mhr = mhr;
		this.proportions = proportions;
		const rest = mhr.skeleton(mhr.jointParameters(this.params(null)));
		this.restTurns = rest.map((g) => unit(g.q));
		const J = JOINTS - FIRST;
		this.names = mhr.names.slice(FIRST);
		this.parents = Array.from({ length: J }, (_, j) => mhr.parents[j + FIRST] - FIRST);
		this.parents[0] = -1;
		// Each joint's offset from its parent at rest, metres; the root's is
		// where it stands at rest.
		this.offsets = Array.from({ length: J }, (_, j) => {
			const t = rest[j + FIRST].t;
			if (j === 0) return [t[0] / 100, t[1] / 100, t[2] / 100];
			const p = rest[this.parents[j] + FIRST].t;
			return [(t[0] - p[0]) / 100, (t[1] - p[1]) / 100, (t[2] - p[2]) / 100];
		});
		this.restRoot = this.offsets[0].slice();
		this.offsets[0] = [0, 0, 0];
		// How high the lowest joint stands at rest over the mesh's soles.
		this.aboveGround = Math.min(...rest.slice(FIRST).map((g) => g.t[1])) / 100;
		// For Kimodo's clean-up and the rigs it retargets onto.
		this.hips = ['r_upleg', 'l_upleg'];
		this.effectors = {
			LeftHand: ['l_wrist', 'l_middle1'], RightHand: ['r_wrist', 'r_middle1'],
			LeftFoot: ['l_foot', 'l_ball'], RightFoot: ['r_foot', 'r_ball'],
		};
	}

	get jointCount() { return this.names.length; }

	index(name) { return this.names.indexOf(name); }

	// A frame's model parameters with the clip's proportions.
	params(modelParams) {
		const p = new Float64Array(MODEL_PARAMS);
		if (modelParams) p.set(modelParams.subarray(0, PROPORTIONS));
		p.set(this.proportions, PROPORTIONS);
		return p;
	}

	// One frame: `local` [J * 4], each joint's parent-local turn (x, y, z, w),
	// and `root` [3], where the pelvis stands in MHR's frame, metres (SAM 3D
	// Body's body-relative position; its camera translation is not added).
	frame(modelParams) {
		const mhr = this.mhr;
		const world = mhr.skeleton(mhr.jointParameters(this.params(modelParams)));
		const J = this.jointCount;
		const turns = new Array(J);
		const local = new Float64Array(J * 4);
		for (let j = 0; j < J; j++) {
			turns[j] = unit(mul(unit(world[j + FIRST].q), conjugate(this.restTurns[j + FIRST])));
			const p = this.parents[j];
			local.set(p < 0 ? turns[j] : unit(mul(conjugate(turns[p]), turns[j])), j * 4);
		}
		const t = world[FIRST].t;
		return { local, root: [t[0] / 100, t[1] / 100, t[2] / 100] };
	}
}
