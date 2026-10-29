// Raw bone names to the anatomical words UniMate was trained on ("thigh_L" ->
// "Left Thigh", "mixamorig:LeftUpLeg" -> "Left Thigh", "joint_18" -> "Bone"):
// a port of the rule-based cleaner (data_process/joint_annotation/
// names_clean_rule.py, clean_joint_name + post_process). The model reads a
// T5 embedding of each cleaned name, so the words must be the ones it saw.

import { VOCAB } from './names_vocab.js';

const V = VOCAB;
const JAPANESE_COMPOUND_NAMES = new Set(V.JAPANESE_COMPOUND_NAMES);
const JAPANESE_COMPOUND_ANIMALS = new Set(V.JAPANESE_COMPOUND_ANIMALS);
const get = (map, key) => (Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined);

const SIDE_TOKEN_RE = /[._]([LR])(?=[._]|\d|$)/;
const SINGLE_SIDE_RE = /^[LR](?:[A-Z][a-z]|[A-Z]{0,2}_[A-Za-z])/;
const TRAILING_TOKEN_RE = /[._](\d+|[LRlr]|x)$/;
const FINGER_CODE_RE = /^[Ff]inger([0-4])\d*(Nub)?$/;
const FINGER_CODE = { 0: 'Thumb', 1: 'Index', 2: 'Middle', 3: 'Ring', 4: 'Pinky' };
const FINGER_SEG_RE = /^[Ff]inger([1-5])(Metacarpal|Proximal|Medial|Distal|Tip)\d*$/;
const FINGER_ORD = { 1: 'Thumb', 2: 'Index', 3: 'Middle', 4: 'Ring', 5: 'Pinky' };
const PAREN_DECOR_RE = /\s*\([^)]*\)/g;
const NAMESPACE_RE = /^[A-Za-z][\w .-]*:/;
const BIP_PREFIX_RE = /^(?:BN_)?Bip\d+(?:[-_ ]+|(?=[LR][A-Z]))/;
const MIXAMORIG_RE = /^mixamorig\d*[:_]/i;

// Python's str.capitalize, str.strip(chars) and friends.
const capitalize = (s) => (s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s);
function stripChars(s, chars, left = true, right = true) {
	let a = 0, b = s.length;
	if (left) while (a < b && chars.includes(s[a])) a++;
	if (right) while (b > a && chars.includes(s[b - 1])) b--;
	return s.slice(a, b);
}
const isUpper = (c) => c !== undefined && c !== c.toLowerCase() && c === c.toUpperCase();
const isLower = (c) => c !== undefined && c !== c.toUpperCase() && c === c.toLowerCase();

function stripTrailingDecorations(name) {
	let side = '';
	for (;;) {
		const m = TRAILING_TOKEN_RE.exec(name);
		if (!m) return [side, name];
		const tok = m[1];
		if (tok === 'L' || tok === 'l') side = 'Left';
		else if (tok === 'R' || tok === 'r') side = 'Right';
		name = name.slice(0, m.index);
	}
}

function canon(token) {
	let value = get(V.CANONICAL, token);
	if (value === undefined && token && token !== capitalize(token)) value = get(V.CANONICAL, capitalize(token));
	return value;
}

function extractSidePrefix(name) {
	let side;
	[side, name] = stripTrailingDecorations(name);
	if (side) {
		const stripped = name.replace(/^(?:Left|Right)(?=[A-Z])|^(?:Left|Right)[_ ]|^[LR][-_]|^[lr]_/, '');
		return [side, stripped || name];
	}
	if (/^[RL][-_.]/.test(name)) return [name[0] === 'L' ? 'Left' : 'Right', name.slice(2)];
	let m = /^([RL]) +/.exec(name);
	if (m) return [m[1] === 'L' ? 'Left' : 'Right', name.slice(m[0].length)];
	m = /^(Left|Right)_/i.exec(name);
	if (m) return [capitalize(m[1]), name.slice(m[0].length)];
	m = /^(Left|Right)(?=[A-Z])/.exec(name);
	if (m) return [m[1], name.slice(m[1].length)];
	if (name.length > 1 && 'LR'.includes(name[0]) && isUpper(name[1])) {
		const t = SIDE_TOKEN_RE.exec(name.slice(1));
		if (t) {
			const at = t.index + 1;
			const rest = stripChars(name.slice(0, at) + name.slice(at + t[0].length), '._');
			return [t[1] === 'L' ? 'Left' : 'Right', rest || name];
		}
		if (SINGLE_SIDE_RE.test(name)) return [name[0] === 'L' ? 'Left' : 'Right', name.slice(1)];
	}
	m = /_([LR])_(\d+)$/.exec(name);
	if (m) return [m[1] === 'L' ? 'Left' : 'Right', name.slice(0, m.index) + '_' + m[2]];
	m = /_([LR])$/.exec(name);
	if (m) return [m[1] === 'L' ? 'Left' : 'Right', name.slice(0, m.index)];
	return ['', name];
}

function splitAndMapTokens(name) {
	const mapped = [];
	for (const t of name.split(/(?=[A-Z])|_/)) {
		if (!t) continue;
		const base = t.replace(/\d+$/, '');
		const value = canon(base);
		if (value !== undefined) {
			if (value) mapped.push(value);
		} else if (base && base.length > 1) {
			mapped.push(base);
		}
	}
	return mapped.join(' ');
}

function withSide(side, result) {
	if (!side) return result;
	const m = /^(Left|Right)\b\s*/.exec(result);
	if (m) result = result.slice(m[0].length);
	return (side + ' ' + result).trim();
}

function cleanJapaneseName(name) {
	let side;
	[side, name] = extractSidePrefix(name);
	const lower = name.replace(/\d+$/, '').toLowerCase();
	const word = get(V.JAPANESE_WORDS_LOWER, lower);
	if (word !== undefined) return withSide(side, word);
	const compound = get(V.PIRRANA_COMPOUNDS, name);
	if (compound !== undefined) return compound;
	return withSide(side, name);
}

function cleanPrefixedName(raw, prefix, map, stripTrailingC = false, skip = []) {
	let name = raw.slice(prefix.length);
	if (stripTrailingC && name.endsWith('_C')) name = name.slice(0, -2);
	let side;
	[side, name] = extractSidePrefix(name);
	for (const sub of skip) if (name.includes(sub)) return '';
	let result = get(map, name);
	if (result === undefined) result = get(map, name.replace(/\d+$/, ''));
	if (result === undefined) result = splitAndMapTokens(name) || name;
	return withSide(side, result).trim();
}

function cleanSpiderName(raw) {
	const hit = get(V.SPIDER_MAP, raw);
	if (hit !== undefined) return hit;
	let m = /^Fang([RL])_(\d+)_/.exec(raw);
	if (m) return `${m[1] === 'R' ? 'Right' : 'Left'} Fang`;
	m = /^Leg_([RL])_(\d)(\d)_/.exec(raw);
	if (m) return `${m[1] === 'R' ? 'Right' : 'Left'} Leg`;
	m = /^_([RL])Toe(\d)_/.exec(raw);
	if (m) return `${m[1] === 'R' ? 'Right' : 'Left'} Leg Tip`;
	return stripChars(raw, '_');
}

function cleanStandardName(raw) {
	let name = raw.replace(NAMESPACE_RE, '') || raw;
	name = name.replace(BIP_PREFIX_RE, '');
	for (const prefix of V.REMOVE_PREFIXES) {
		if (name.startsWith(prefix)) {
			name = name.slice(prefix.length);
			break;
		}
	}
	name = stripChars(name, '_', true, false).trim();
	if (!name) return stripChars(raw, '_');
	let side;
	[side, name] = extractSidePrefix(name);
	let m = FINGER_SEG_RE.exec(name);
	if (m) return withSide(side, FINGER_ORD[m[1]] + ' Finger' + (m[2] === 'Tip' ? ' End' : ''));
	m = FINGER_CODE_RE.exec(name);
	if (m) return withSide(side, FINGER_CODE[m[1]] + ' Finger' + (m[2] ? ' End' : ''));
	m = /^(.+?)_?(\d+)$/.exec(name);
	const base = m ? stripChars(m[1], '_', false, true) : name;
	const mappedBase = canon(base);
	if (mappedBase !== undefined) {
		if (!mappedBase || mappedBase === 'Bone') return 'Bone';
		return withSide(side, mappedBase);
	}
	const mapped = [];
	for (const p of base.split('_').filter(Boolean)) {
		if (p === 'L' || p === 'l') { side = side || 'Left'; continue; }
		if (p === 'R' || p === 'r') { side = side || 'Right'; continue; }
		const subBase = p.replace(/\d+$/, '');
		const subMapped = canon(subBase);
		if (subMapped !== undefined) {
			if (subMapped) mapped.push(subMapped);
		} else if (subBase) {
			for (const st of subBase.split(/(?=[A-Z])/).filter(Boolean)) {
				const stMapped = canon(st);
				if (stMapped !== undefined) {
					if (stMapped) mapped.push(stMapped);
				} else if (st.length > 1) {
					mapped.push(st);
				}
			}
		}
	}
	const result = mapped.join(' ');
	if (!result || result === 'Bone') return 'Bone';
	return withSide(side, result).trim() || raw;
}

export function cleanJointName(raw, animal = '') {
	if (!raw || !raw.trim()) return raw;
	raw = raw.replace(PAREN_DECOR_RE, '').trim() || raw;
	if (/^Bone\d+$/.test(raw)) return 'Bone';
	if (/^_?\d+$/.test(stripChars(raw, '_'))) return raw;
	const mix = MIXAMORIG_RE.exec(raw);
	if (mix) return cleanPrefixedName(raw, mix[0], V.MIXAMO_MAP);
	let hit = get(V.STANDALONE_MAP, raw);
	if (hit !== undefined) return hit;
	hit = get(V.SABRECAT_MAP, raw);
	if (hit !== undefined) return hit;
	if (animal === 'Spider') return cleanSpiderName(raw);
	if (raw.startsWith('Sabrecat')) return get(V.SABRECAT_MAP, raw) ?? raw;
	if (raw.startsWith('NPC_')) return cleanPrefixedName(raw, 'NPC_', V.NPC_DIRECT, false, ['Jiggle']);
	if (raw.startsWith('Elk')) return cleanPrefixedName(raw, 'Elk', V.ELK_MAP);
	if (raw.startsWith('jt_')) return cleanPrefixedName(raw, 'jt_', V.JT_MAP, true);
	const baseLower = raw.replace(/^[RL]_/, '').toLowerCase().replace(/\d+$/, '');
	if (get(V.JAPANESE_WORDS_LOWER, baseLower) !== undefined) return cleanJapaneseName(raw);
	if (JAPANESE_COMPOUND_ANIMALS.has(animal) && JAPANESE_COMPOUND_NAMES.has(raw)) return cleanJapaneseName(raw);
	return cleanStandardName(raw);
}

export function postProcess(name) {
	name = stripChars(name.replace(/\s+/g, ' ').trim(), '._ ', false, true);
	name = name.replace(/\s+\d+$/, '');
	name = name.replace(/(\w)\d+\s+End/g, '$1 End');
	name = name.replace(/\bHand (Thumb|Index|Middle|Ring|Pinky) Finger\b/g, '$1 Finger');
	name = name.replace(/\b(Thumb|Index|Middle|Ring|Pinky) Finger Finger\b/g, '$1 Finger');
	name = name.replace(/\bFinger (Thumb|Index|Middle|Ring|Pinky) Finger\b/g, '$1 Finger');
	name = name.replace(/\b(?:Up|Upper) Leg$/, 'Thigh');
	name = name.replace(/\bLower Leg$/, 'Shin');
	name = name.replace(/\b(?:Lower|Fore) Arm$/, 'Forearm');
	name = name.replace(/\bToe Base$/, 'Toe');
	name = name.replace(/^(?!Left |Right )(.+) (Left|Right)$/, '$2 $1');
	name = name.replace(/^Head Top End$/, 'Head End');
	name = name.split('. ').join(' ');
	name = name.replace(/^Spine ?\d* ?Tail$/, 'Tail');
	name = name.replace(/^Head ?\d* ?Jaw$/, 'Jaw');
	name = name.replace(/^Head ?\d* ?Eyelid$/, 'Eyelid');
	name = name.replace(/^Head Muzzle$/, 'Muzzle');
	name = name.replace(/^Head Jaw End$/, 'Jaw End');
	name = name.replace(/^Head Brain$/, 'Head');
	name = name.replace(/Bip \d+ /g, '');
	name = name.replace(/^(\w+) \1$/, '$1');
	if (/^Xtra/.test(name)) name = 'Bone';
	name = name.replace(/Ponytail\d*.*/, 'Appendage');
	name = name.split(/\s+/).filter(Boolean).map((w) => (isLower(w[0]) && w.length > 1 ? capitalize(w) : w)).join(' ');
	name = name.replace(/^Spine (Left|Right) Wing$/, '$1 Wing');
	name = name.replace(/\s+\d+$/, '');
	return name || 'Bone';
}

// The name the model is conditioned on.
export function modelJointName(raw, animal = '') {
	return postProcess(cleanJointName(raw, animal));
}
