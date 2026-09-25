// Colours and sizes, in one place. The palette is examples/editor.js's, so the
// studio reads as the same family of tool.

export const THEME = {
	panel: [0.19, 0.19, 0.19],
	header: [0.16, 0.16, 0.16],
	body: [0.11, 0.11, 0.11],
	canvas: [0.085, 0.085, 0.09],
	grid: [1, 1, 1, 0.035],
	accent: [0.28, 0.45, 0.70],
	text: [0.90, 0.90, 0.90],
	dim: [0.62, 0.62, 0.62],
	border: [1, 1, 1, 0.08],
	hover: [0.24, 0.24, 0.24],
	node: [0.17, 0.17, 0.18],
	nodeHeader: [0.23, 0.23, 0.25],
	running: [0.35, 0.75, 0.40],
	error: [0.90, 0.35, 0.30],
	cached: [0.45, 0.45, 0.50],
	radius: 4,
};

// Port colours by type, so a wire says what it carries at a glance.
export const TYPE_COLOR = {
	IMAGE: [0.39, 0.71, 0.96],
	LATENT: [1.0, 0.62, 0.80],
	CONDITIONING: [1.0, 0.66, 0.19],
	MODEL: [0.70, 0.62, 0.86],
	TEXT_ENCODER: [1.0, 0.84, 0.0],
	VAE: [1.0, 0.43, 0.43],
	LORA: [0.49, 0.86, 0.55],
	REFERENCES: [0.30, 0.82, 0.88],
	ANY: [0.8, 0.8, 0.8],
};
export const PRIMITIVE_COLOR = [0.60, 0.60, 0.62];

export function typeColor(type) { return TYPE_COLOR[type] ?? PRIMITIVE_COLOR; }

export const BAR_H = 28;       // cui's menu bar
export const STATUS_H = 30;
export const PALETTE_W = 230;
export const INSPECTOR_W = 330;

// Node geometry, in graph units (one unit is one point at zoom 1).
export const NODE_W = 280;
export const NODE_HEAD = 26;
export const NODE_ROW = 22;
export const NODE_PAD = 8;
export const PORT_R = 5;

// The editors inside a node: inset from its sides, a small label over each box,
// and the gap between them.
export const FIELD_X = 12;
export const FIELD_H = 26;
export const LABEL_H = 14;
export const PROMPT_H = 90;
export const FIELD_GAP = 6;
// Below this zoom the editors are too small to use, and a node shows its values
// as text instead.
export const EDIT_ZOOM = 0.5;

// How much one wheel notch zooms the canvas.
export const ZOOM_STEP = 1.1;
