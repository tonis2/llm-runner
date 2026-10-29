// SAM 3D Body's 70 keypoints (MHR70), from its `metadata/mhr70.py`: the names,
// in order, and the 65 links its skeleton drawing joins; and that drawing.

export const KEYPOINT_NAMES = [
	'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
	'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee',
	'right_knee', 'left_ankle', 'right_ankle', 'left_big_toe', 'left_small_toe', 'left_heel',
	'right_big_toe', 'right_small_toe', 'right_heel', 'right_thumb4', 'right_thumb3', 'right_thumb2',
	'right_thumb_third_joint', 'right_forefinger4', 'right_forefinger3', 'right_forefinger2', 'right_forefinger_third_joint', 'right_middle_finger4',
	'right_middle_finger3', 'right_middle_finger2', 'right_middle_finger_third_joint', 'right_ring_finger4', 'right_ring_finger3', 'right_ring_finger2',
	'right_ring_finger_third_joint', 'right_pinky_finger4', 'right_pinky_finger3', 'right_pinky_finger2', 'right_pinky_finger_third_joint', 'right_wrist',
	'left_thumb4', 'left_thumb3', 'left_thumb2', 'left_thumb_third_joint', 'left_forefinger4', 'left_forefinger3',
	'left_forefinger2', 'left_forefinger_third_joint', 'left_middle_finger4', 'left_middle_finger3', 'left_middle_finger2', 'left_middle_finger_third_joint',
	'left_ring_finger4', 'left_ring_finger3', 'left_ring_finger2', 'left_ring_finger_third_joint', 'left_pinky_finger4', 'left_pinky_finger3',
	'left_pinky_finger2', 'left_pinky_finger_third_joint', 'left_wrist', 'left_olecranon', 'right_olecranon', 'left_cubital_fossa',
	'right_cubital_fossa', 'left_acromion', 'right_acromion', 'neck',
];

export const KEYPOINT_LINKS = [
	[13, 11], [11, 9], [14, 12], [12, 10], [9, 10], [5, 9], [6, 10], [5, 6], [5, 7], [6, 8],
	[7, 62], [8, 41], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [13, 15],
	[13, 16], [13, 17], [14, 18], [14, 19], [14, 20], [62, 45], [45, 44], [44, 43], [43, 42], [62, 49],
	[49, 48], [48, 47], [47, 46], [62, 53], [53, 52], [52, 51], [51, 50], [62, 57], [57, 56], [56, 55],
	[55, 54], [62, 61], [61, 60], [60, 59], [59, 58], [41, 24], [24, 23], [23, 22], [22, 21], [41, 28],
	[28, 27], [27, 26], [26, 25], [41, 32], [32, 31], [31, 30], [30, 29], [41, 36], [36, 35], [35, 34],
	[34, 33], [41, 40], [40, 39], [39, 38], [38, 37],
];

// The keypoints and links over a copy of the image, RGB. `pose` is a pose
// (pred_keypoints_2d) or a posed frame (keypoints_2d).
export function drawPreview(img, pose) {
	const { width, height, channels } = img;
	const px = new Uint8Array(width * height * 3);
	for (let i = 0; i < width * height; i++) for (let c = 0; c < 3; c++) px[i * 3 + c] = img.pixels[i * channels + c];
	const r = Math.max(2, Math.round(Math.max(width, height) / 300));
	const dot = (x, y, rad, col) => {
		for (let dy = -rad; dy <= rad; dy++) {
			for (let dx = -rad; dx <= rad; dx++) {
				const X = Math.round(x) + dx, Y = Math.round(y) + dy;
				if (dx * dx + dy * dy > rad * rad || X < 0 || Y < 0 || X >= width || Y >= height) continue;
				px.set(col, (Y * width + X) * 3);
			}
		}
	};
	const kp = pose.pred_keypoints_2d ?? pose.keypoints_2d;
	for (const [a, b] of KEYPOINT_LINKS) {
		const x0 = kp[a * 2], y0 = kp[a * 2 + 1], x1 = kp[b * 2], y1 = kp[b * 2 + 1];
		const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0));
		for (let s = 0; s <= steps; s++) dot(x0 + ((x1 - x0) * s) / steps, y0 + ((y1 - y0) * s) / steps, Math.max(1, r >> 1), [40, 220, 90]);
	}
	for (let k = 0; k < kp.length / 2; k++) dot(kp[k * 2], kp[k * 2 + 1], r, [255, 60, 40]);
	return { width, height, channels: 3, pixels: px };
}
