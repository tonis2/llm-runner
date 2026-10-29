// The skeletons Kimodo's motion checkpoints predict, keyed by the motion GGUF's
// `kimodo.skeleton`. Joint names and parents are NVIDIA Kimodo's
// (kimodo/skeleton/definitions.py, Apache-2.0); the parent-local rest offsets,
// in metres, are the ones kimodo.cpp extracted from its joints.p assets.
//
// A motion frame of a J-joint skeleton is 9 + 12 J floats: the global root
// (x, y, z, heading cos, sin), then the body - J joint positions, J 6D global
// rotations and the rest (velocities, foot contacts) that decoding does not read.
//
// `hips` are the right and left hip joints the facing is read from, and
// `effectors` each hand and foot as a chain - the joint itself, then the one
// beyond it - the way upstream's end-effector constraints name them.

export const SKELETONS = {
	soma30: {
		hips: ['RightLeg', 'LeftLeg'],
		effectors: {
			LeftHand: ['LeftHand', 'LeftHandMiddleEnd'], RightHand: ['RightHand', 'RightHandMiddleEnd'],
			LeftFoot: ['LeftFoot', 'LeftToeBase'], RightFoot: ['RightFoot', 'RightToeBase'],
		},
		names: ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw',
			'LeftEye', 'RightEye', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
			'LeftHandThumbEnd', 'LeftHandMiddleEnd', 'RightShoulder', 'RightArm', 'RightForeArm',
			'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd', 'LeftLeg', 'LeftShin', 'LeftFoot',
			'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase'],
		parents: [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28],
		offsets: [
			[0, 0, 0], [-0.00013727, 0.0500376256, -0.00053726669], [-1.86574103e-9, 0.0712530139, -0.000298248546],
			[-5.75188398e-9, 0.0755006305, -0.00815970992], [-0.00181676517, 0.263112953, -0.00553348292],
			[-2.85102231e-8, 0.0770939664, 0.0230258546], [-4.5975437e-8, 0.0612891595, 0.0195370861],
			[2.63687901e-5, 0.0047559225, 0.0309494062], [0.0320638079, 0.0538020513, 0.0758688308],
			[-0.0322244017, 0.05361869, 0.0755823359], [0.0162165175, 0.232371641, 0.0511341324],
			[0.149198457, 2.19397873e-8, -0.0550232576], [0.287393078, 2.50268389e-9, -2.58787737e-5],
			[0.270939812, -7.06625108e-9, 2.60897248e-5], [0.122686267, -0.0322017573, 0.0483306876],
			[0.190119595, -0.00312878387, -0.000339570373], [-0.0138011824, 0.231803086, 0.0521415786],
			[-0.150371962, 1.17387901e-7, -0.0554560437], [-0.287366393, 1.87628082e-8, -2.59709359e-5],
			[-0.271336198, -1.16767401e-9, 2.61269368e-5], [-0.122642483, -0.0321145448, 0.0480403904],
			[-0.190005945, -0.00306615542, -0.0003157343], [0.10043214, -0.0843452671, 0.0259565473],
			[-1e-8, -0.432217537, -0.00802912805], [1e-8, -0.421550959, -0.0348152298],
			[0, -0.0505947206, 0.132315294], [-0.10047278, -0.0829525995, 0.0262031695],
			[1e-8, -0.433622059, -0.00805555828], [2e-8, -0.421173943, -0.0347839785],
			[-3.42907669e-9, -0.0507960932, 0.132841956]],
	},
	smplx22: {
		hips: ['right_hip', 'left_hip'],
		effectors: {
			LeftHand: ['left_wrist'], RightHand: ['right_wrist'],
			LeftFoot: ['left_ankle', 'left_foot'], RightFoot: ['right_ankle', 'right_foot'],
		},
		names: ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
			'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
			'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
			'right_elbow', 'left_wrist', 'right_wrist'],
		parents: [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
		offsets: [
			[0, 0, 0], [0.052299179, -0.093935639, -0.027606763], [-0.057192899, -0.10654819, -0.022217851],
			[-0.001495834, 0.11292994, -0.024981268], [0.058866613, -0.416441321, -0.006556974],
			[-0.048074268, -0.397559673, -0.014061437], [0.006900469, 0.145636231, -0.00685851],
			[-0.041737989, -0.437583506, -0.029511765], [0.014489345, -0.446852267, -0.018029511],
			[-0.010334037, 0.056081813, 0.021115851], [0.04929354, -0.065279245, 0.126259089],
			[-0.040575184, -0.065286517, 0.127075911], [-0.011025756, 0.171365142, -0.028827066],
			[0.047724526, 0.087643057, -0.00837545], [-0.046636276, 0.086612143, -0.014864366],
			[0.024654359, 0.175390735, 0.024463326], [0.126284808, 0.057680372, -0.013885141],
			[-0.109341696, 0.053674292, -0.00911788], [0.272907287, -0.069853373, -0.039094493],
			[-0.292028785, -0.035440356, -0.024564851], [0.27617383, 0.021254137, -0.00247822],
			[-0.271878421, -0.004834589, -0.016445294]],
	},
	g1skel34: {
		hips: ['right_hip_pitch_skel', 'left_hip_pitch_skel'],
		effectors: {
			LeftHand: ['left_wrist_yaw_skel', 'left_hand_roll_skel'], RightHand: ['right_wrist_yaw_skel', 'right_hand_roll_skel'],
			LeftFoot: ['left_ankle_roll_skel', 'left_toe_base'], RightFoot: ['right_ankle_roll_skel', 'right_toe_base'],
		},
		names: ['pelvis_skel', 'left_hip_pitch_skel', 'left_hip_roll_skel', 'left_hip_yaw_skel',
			'left_knee_skel', 'left_ankle_pitch_skel', 'left_ankle_roll_skel', 'left_toe_base',
			'right_hip_pitch_skel', 'right_hip_roll_skel', 'right_hip_yaw_skel', 'right_knee_skel',
			'right_ankle_pitch_skel', 'right_ankle_roll_skel', 'right_toe_base', 'waist_yaw_skel',
			'waist_roll_skel', 'waist_pitch_skel', 'left_shoulder_pitch_skel', 'left_shoulder_roll_skel',
			'left_shoulder_yaw_skel', 'left_elbow_skel', 'left_wrist_roll_skel', 'left_wrist_pitch_skel',
			'left_wrist_yaw_skel', 'left_hand_roll_skel', 'right_shoulder_pitch_skel',
			'right_shoulder_roll_skel', 'right_shoulder_yaw_skel', 'right_elbow_skel',
			'right_wrist_roll_skel', 'right_wrist_pitch_skel', 'right_wrist_yaw_skel', 'right_hand_roll_skel'],
		parents: [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32],
		offsets: [
			[0, 0, 0], [0.064452, -0.1027, 0], [0.052, -0.030465, 0], [0, -0.12412, 0.025001],
			[0.0021489, -0.17734, -0.078273], [-0.000094445, -0.30001, 0], [0, -0.017558, 0], [0, -0.035, 0.14],
			[-0.064452, -0.1027, 0], [-0.052, -0.030465, 0], [0, -0.12412, 0.025001], [-0.0021489, -0.17734, -0.078273],
			[0.000094445, -0.30001, 0], [0, -0.017558, 0], [0, -0.035, 0.14], [0, 0, 0], [0, 0.044, -0.0039635],
			[0, 0, 0], [0.10022, 0.24778, 0.0039563], [0.038, -0.013831, 0], [0.00624, -0.1032, 0],
			[0, -0.080518, 0.015783], [0.00188791, -0.01, 0.1], [0, 0, 0.038], [0, 0, 0.046], [0, 0, 0.1],
			[-0.10021, 0.24778, 0.0039563], [-0.038, -0.013831, 0], [-0.00624, -0.1032, 0],
			[0, -0.080518, 0.015783], [-0.00188791, -0.01, 0.1], [0, 0, 0.038], [0, 0, 0.046], [0, 0, 0.1]],
	},
};
