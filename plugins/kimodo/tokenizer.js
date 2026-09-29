// Kimodo's Llama-3 byte-level BPE, read from its `tokenizer.gguf`.
//
// That file keeps the vocabulary under `kimodo.tokenizer.*` rather than the
// `tokenizer.ggml.*` keys the host tokenizer reads, so the tokenizer is here:
// the Llama-3 pre-split, then BPE merges by rank, as kimodo.cpp's
// `llm_tokenizer.cpp` does it (letters are ASCII letters and every byte >= 0x80,
// the way that one approximates the Llama-3 regex). A prompt's ids start with
// BOS; nothing else is added.

import { llm } from '../lib/llm.js';

const SEPARATOR = '\x1f';

function utf8Bytes(text) {
	const out = [];
	for (const ch of text) {
		const c = ch.codePointAt(0);
		if (c < 0x80) out.push(c);
		else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
		else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
		else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 0x3f), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
	}
	return out;
}

const isAlpha = (b) => (b >= 65 && b <= 90) || (b >= 97 && b <= 122) || b >= 0x80;
const isDigit = (b) => b >= 48 && b <= 57;
const isSpace = (b) => b === 32 || b === 9 || b === 13 || b === 10;
const CONTRACTIONS = ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"].map((s) => Array.from(s, (c) => c.charCodeAt(0)));

function contraction(bytes, at) {
	if (bytes[at] !== 39) return 0;
	for (const option of CONTRACTIONS) {
		if (at + option.length > bytes.length) continue;
		let ok = true;
		for (let i = 1; i < option.length; i++) {
			const b = bytes[at + i];
			const lower = b >= 65 && b <= 90 ? b + 32 : b;
			if (lower !== option[i]) { ok = false; break; }
		}
		if (ok) return option.length;
	}
	return 0;
}

// The prompt's bytes cut into words, as [start, end) pairs.
function preSplit(bytes) {
	const words = [];
	const n = bytes.length;
	let pos = 0;
	while (pos < n) {
		let size = contraction(bytes, pos);
		if (size) { words.push([pos, pos + size]); pos += size; continue; }
		const first = bytes[pos];
		const prefixed = !isDigit(first) && first !== 13 && first !== 10 && !isAlpha(first) && pos + 1 < n && isAlpha(bytes[pos + 1]);
		if (isAlpha(first) || prefixed) {
			size = prefixed ? 1 : 0;
			while (pos + size < n && isAlpha(bytes[pos + size])) size++;
		} else if (isDigit(first)) {
			while (size < 3 && pos + size < n && isDigit(bytes[pos + size])) size++;
		} else if (!isSpace(first) || (pos + 1 < n && !isSpace(bytes[pos + 1]) && !isAlpha(bytes[pos + 1]) && !isDigit(bytes[pos + 1]))) {
			size = first === 32 ? 1 : 0;
			while (pos + size < n && !isSpace(bytes[pos + size]) && !isAlpha(bytes[pos + size]) && !isDigit(bytes[pos + size])) size++;
		} else {
			while (pos + size < n && isSpace(bytes[pos + size])) size++;
		}
		words.push([pos, pos + size]);
		pos += size;
	}
	return words;
}

export class KimodoTokenizer {
	constructor(path) {
		const m = llm.open(path);
		if (m.meta('general.architecture') !== 'kimodo-llm2vec-tokenizer') {
			m.close();
			throw new Error(`${path} is not a Kimodo LLM2Vec tokenizer GGUF`);
		}
		const tokens = m.array('kimodo.tokenizer.tokens');
		const merges = m.array('kimodo.tokenizer.merges');
		this.bos = m.meta('kimodo.bos_token_id', 128000);
		m.close();
		this.ids = new Map();
		tokens.forEach((t, i) => this.ids.set(t, i));
		this.ranks = new Map();
		merges.forEach((merge, i) => {
			const split = merge.indexOf(' ');
			this.ranks.set(merge.slice(0, split) + SEPARATOR + merge.slice(split + 1), i);
		});
		// GPT-2's printable stand-in for each byte.
		this.byteChar = new Array(256);
		let extra = 256;
		for (let b = 0; b < 256; b++) {
			const direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
			this.byteChar[b] = String.fromCodePoint(direct ? b : extra++);
		}
	}

	// Token ids, BOS first.
	encode(text) {
		const bytes = utf8Bytes(text);
		const out = [this.bos];
		for (const [start, end] of preSplit(bytes)) {
			const symbols = [];
			for (let i = start; i < end; i++) symbols.push(this.byteChar[bytes[i]]);
			while (symbols.length > 1) {
				let bestRank = Infinity, best = -1;
				for (let i = 0; i + 1 < symbols.length; i++) {
					const rank = this.ranks.get(symbols[i] + SEPARATOR + symbols[i + 1]);
					if (rank !== undefined && rank < bestRank) { bestRank = rank; best = i; }
				}
				if (best < 0) break;
				symbols.splice(best, 2, symbols[best] + symbols[best + 1]);
			}
			for (const s of symbols) {
				const id = this.ids.get(s);
				if (id === undefined) throw new Error(`BPE symbol ${JSON.stringify(s)} is not in the vocabulary`);
				out.push(id);
			}
		}
		return Uint32Array.from(out);
	}
}
