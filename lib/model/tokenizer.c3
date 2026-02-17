module llm;

import std::io;
import std::core::mem;
import std::ascii;

const uint TOKEN_NORMAL  = 1;
const uint TOKEN_CONTROL = 3;
const uint TOKEN_BYTE    = 6;

// --- Vocab hash map (string -> id + score) ---

fn usz next_pow2(usz n) {
    usz p = 1;
    while (p < n) p *= 2;
    return p;
}

struct VocabMap {
    String[] keys;
    uint[] ids;
    float[] scores;
    bool[] occupied;
    usz capacity;
}

fn VocabMap create_vocab_map(String[] vocab, float[] scores, uint[] token_types) {
    VocabMap map;
    usz cap = next_pow2(vocab.len * 2);
    map.capacity = cap;
    map.keys = mem::new_array(String, cap);
    map.ids = mem::new_array(uint, cap);
    map.scores = mem::new_array(float, cap);
    map.occupied = mem::new_array(bool, cap);

    for (usz i = 0; i < cap; i++) {
        map.occupied[i] = false;
    }

    usz mask = cap - 1;
    for (usz i = 0; i < vocab.len; i++) {
        // Only index NORMAL tokens for BPE lookups
        if (token_types[i] == TOKEN_CONTROL || token_types[i] == TOKEN_BYTE) continue;
        usz idx = vocab[i].hash() & mask;
        while (map.occupied[idx]) {
            idx = (idx + 1) & mask;
        }
        map.keys[idx] = vocab[i];
        map.ids[idx] = (uint)i;
        map.scores[idx] = scores[i];
        map.occupied[idx] = true;
    }

    return map;
}

fn bool VocabMap.lookup(&self, String key, uint* out_id, float* out_score) {
    usz mask = self.capacity - 1;
    usz idx = key.hash() & mask;
    usz start = idx;
    while (self.occupied[idx]) {
        if (self.keys[idx] == key) {
            *out_id = self.ids[idx];
            *out_score = self.scores[idx];
            return true;
        }
        idx = (idx + 1) & mask;
        if (idx == start) break;
    }
    return false;
}

// Lookup by ID only (no score needed)
fn bool VocabMap.lookup_id(&self, String key, uint* out_id) {
    float dummy;
    return self.lookup(key, out_id, &dummy);
}

fn void VocabMap.free_map(&self) {
    if (self.keys.len > 0) mem::free(self.keys);
    if (self.ids.len > 0) mem::free(self.ids);
    if (self.scores.len > 0) mem::free(self.scores);
    if (self.occupied.len > 0) mem::free(self.occupied);
}

// --- Merge map for BPE (pair of strings -> rank) ---

const uint MERGE_NOT_FOUND = 0xFFFFFFFF;

struct MergeMap {
    String[] keys_a;
    String[] keys_b;
    uint[] ranks;
    bool[] occupied;
    usz capacity;
}

fn usz merge_hash(String a, String b) {
    usz h = 14695981039346656037;
    for (usz i = 0; i < a.len; i++) {
        h ^= (usz)a[i];
        h *= 1099511628211;
    }
    h ^= 0xFF;
    h *= 1099511628211;
    for (usz i = 0; i < b.len; i++) {
        h ^= (usz)b[i];
        h *= 1099511628211;
    }
    return h;
}

fn MergeMap create_merge_map(usz expected_count) {
    MergeMap map;
    usz cap = next_pow2(expected_count * 2);
    if (cap < 1024) cap = 1024;
    map.capacity = cap;
    map.keys_a = mem::new_array(String, cap);
    map.keys_b = mem::new_array(String, cap);
    map.ranks = mem::new_array(uint, cap);
    map.occupied = mem::new_array(bool, cap);
    for (usz i = 0; i < cap; i++) {
        map.occupied[i] = false;
    }
    return map;
}

fn void MergeMap.insert(&self, String a, String b, uint rank) {
    usz mask = self.capacity - 1;
    usz idx = merge_hash(a, b) & mask;
    while (self.occupied[idx]) {
        idx = (idx + 1) & mask;
    }
    self.keys_a[idx] = a;
    self.keys_b[idx] = b;
    self.ranks[idx] = rank;
    self.occupied[idx] = true;
}

fn uint MergeMap.lookup(&self, String a, String b) {
    usz mask = self.capacity - 1;
    usz idx = merge_hash(a, b) & mask;
    usz start = idx;
    while (self.occupied[idx]) {
        if (self.keys_a[idx] == a && self.keys_b[idx] == b) {
            return self.ranks[idx];
        }
        idx = (idx + 1) & mask;
        if (idx == start) break;
    }
    return MERGE_NOT_FOUND;
}

fn void MergeMap.free_map(&self) {
    if (self.keys_a.len > 0) mem::free(self.keys_a);
    if (self.keys_b.len > 0) mem::free(self.keys_b);
    if (self.ranks.len > 0) mem::free(self.ranks);
    if (self.occupied.len > 0) mem::free(self.occupied);
}

// --- Tokenizer ---

struct Tokenizer {
    String[] vocab;
    uint[] token_types;
    float[] scores;
    uint vocab_size;
    VocabMap vocab_map;
    uint[256] byte_tokens;
    bool is_bpe;
    bool ignore_merges;
    uint bos_id;
    uint eos_id;
    MergeMap merge_map;
}

fn int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

fn Tokenizer? load_tokenizer(GGUFFile* gf) {
    String[] vocab = gf.read_string_array("tokenizer.ggml.tokens")!;
    uint[] token_types = gf.read_u32_array("tokenizer.ggml.token_type")!;

    uint vocab_size = (uint)vocab.len;

    // Detect tokenizer type
    String tok_model = gf.get_string("tokenizer.ggml.model") ?? "llama";
    bool is_bpe = tok_model == "gpt2";

    // Read scores (may not exist for BPE models)
    float[] scores;
    if (try scores_result = gf.read_f32_array("tokenizer.ggml.scores")) {
        scores = scores_result;
    } else {
        // BPE models may not have scores; allocate zeros
        scores = mem::new_array(float, vocab_size);
        for (usz i = 0; i < vocab_size; i++) scores[i] = 0.0f;
    }

    VocabMap vocab_map = create_vocab_map(vocab, scores, token_types);

    // Build byte token lookup: byte value -> token id
    uint[256] byte_tokens;
    for (usz i = 0; i < 256; i++) byte_tokens[i] = 0;
    for (usz i = 0; i < vocab.len; i++) {
        if (token_types[i] == TOKEN_BYTE && vocab[i].len == 6) {
            int hi = hex_val(vocab[i][3]);
            int lo = hex_val(vocab[i][4]);
            if (hi >= 0 && lo >= 0) {
                byte_tokens[(usz)(hi * 16 + lo)] = (uint)i;
            }
        }
    }

    // Read BOS/EOS from GGUF metadata, default to 1/2 (SPM convention)
    uint bos_id = gf.get_u32("tokenizer.ggml.bos_token_id") ?? 1;
    uint eos_id = gf.get_u32("tokenizer.ggml.eos_token_id") ?? 2;

    // Load merges for BPE
    MergeMap merge_map;
    bool ignore_merges = false;
    if (is_bpe) {
        if (try merges_result = gf.read_string_array("tokenizer.ggml.merges")) {
            merge_map = create_merge_map(merges_result.len);
            for (usz i = 0; i < merges_result.len; i++) {
                // Each merge line is "token_a token_b" separated by space
                String line = merges_result[i];
                if (try split = line.index_of_char(' ')) {
                    if (split > 0 && split < line.len - 1) {
                        String a = (String)line[0..split - 1];
                        String b = (String)line[split + 1..line.len - 1];
                        merge_map.insert(a, b, (uint)i);
                    }
                }
            }
            io::printfn("  Loaded %d BPE merges", merges_result.len);
            mem::free(merges_result);
        } else {
            merge_map = create_merge_map(0);
        }

        // Check ignore_merges flag (LLaMA 3 sets this)
        // In GGUF this is stored as tokenizer.ggml.ignore_merges (bool)
        // If not present, we can heuristically enable it for large vocab BPE models
        ignore_merges = vocab_size >= 100000;
    }

    io::printfn("Tokenizer loaded: %d tokens (%s), BOS=%d, EOS=%d",
        vocab_size, is_bpe ? "BPE" : "SPM", bos_id, eos_id);

    return {
        .vocab = vocab,
        .token_types = token_types,
        .scores = scores,
        .vocab_size = vocab_size,
        .vocab_map = vocab_map,
        .byte_tokens = byte_tokens,
        .is_bpe = is_bpe,
        .ignore_merges = ignore_merges,
        .bos_id = bos_id,
        .eos_id = eos_id,
        .merge_map = merge_map,
    };
}

// --- Decode (token ID -> text) ---

// GPT-2 byte encoding: maps 256 byte values to unicode codepoints.
// Pass-through: 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF map to themselves.
// Remapped: 0x00-0x20, 0x7F-0xA0, 0xAD map to U+0100..U+0143.
// This reverses: codepoint -> original byte value, or -1 if unknown.
fn int gpt2_byte_decode(uint cp) {
    if (cp >= 0x21 && cp <= 0x7E) return (int)cp;
    if (cp >= 0xA1 && cp <= 0xAC) return (int)cp;
    if (cp >= 0xAE && cp <= 0xFF) return (int)cp;
    if (cp >= 0x100 && cp <= 0x143) {
        uint n = cp - 0x100;
        if (n <= 32) return (int)n;           // 0x00..0x20
        if (n <= 66) return (int)(n - 33 + 0x7F); // 0x7F..0xA0
        return 0xAD;                          // n == 67
    }
    return -1;
}

// GPT-2 byte encoding: maps raw byte values to unicode codepoints.
// This is the reverse of gpt2_byte_decode.
// Pass-through: 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF map to themselves.
// Remapped: 0x00-0x20 -> U+0100..U+0120, 0x7F-0xA0 -> U+0121..U+0142, 0xAD -> U+0143.
fn uint gpt2_byte_encode(char byte_val) {
    uint b = (uint)byte_val;
    if (b >= 0x21 && b <= 0x7E) return b;
    if (b >= 0xA1 && b <= 0xAC) return b;
    if (b >= 0xAE && b <= 0xFF) return b;
    if (b <= 0x20) return 0x100 + b;
    if (b >= 0x7F && b <= 0xA0) return 0x100 + 33 + (b - 0x7F);
    return 0x143; // 0xAD
}

// Convert raw bytes to GPT-2 byte-encoded UTF-8 string.
// Each input byte maps to a unicode codepoint, then encoded as UTF-8.
// Returns the number of bytes written to buf.
fn usz gpt2_encode_to_buf(String raw, char[] buf) {
    usz out = 0;
    for (usz i = 0; i < raw.len; i++) {
        uint cp = gpt2_byte_encode(raw[i]);
        if (cp < 0x80) {
            buf[out] = (char)cp;
            out++;
        } else if (cp < 0x800) {
            buf[out] = (char)(0xC0 | (cp >> 6));
            buf[out + 1] = (char)(0x80 | (cp & 0x3F));
            out += 2;
        } else {
            buf[out] = (char)(0xE0 | (cp >> 12));
            buf[out + 1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            buf[out + 2] = (char)(0x80 | (cp & 0x3F));
            out += 3;
        }
    }
    return out;
}

fn String Tokenizer.decode_token(&self, uint token_id, char[] buf) {
    if (token_id >= self.vocab_size) return "";

    uint ttype = self.token_types[token_id];
    if (ttype == TOKEN_CONTROL) return "";

    String piece = self.vocab[token_id];

    if (ttype == TOKEN_BYTE) {
        if (piece.len == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
            int hi = hex_val(piece[3]);
            int lo = hex_val(piece[4]);
            if (hi >= 0 && lo >= 0) {
                buf[0] = (char)(hi * 16 + lo);
                return (String)buf[0..0];
            }
        }
        return "";
    }

    // BPE tokens use GPT-2 byte encoding — decode back to raw bytes
    if (self.is_bpe) {
        usz out = 0;
        usz i = 0;
        while (i < piece.len) {
            // Decode UTF-8 codepoint
            uint cp;
            usz cplen;
            char b0 = piece[i];
            if ((b0 & 0x80) == 0) {
                cp = (uint)b0;
                cplen = 1;
            } else if ((b0 & 0xE0) == 0xC0 && i + 1 < piece.len) {
                cp = ((uint)(b0 & 0x1F) << 6) | (uint)(piece[i + 1] & 0x3F);
                cplen = 2;
            } else if ((b0 & 0xF0) == 0xE0 && i + 2 < piece.len) {
                cp = ((uint)(b0 & 0x0F) << 12) | ((uint)(piece[i + 1] & 0x3F) << 6) | (uint)(piece[i + 2] & 0x3F);
                cplen = 3;
            } else {
                cp = (uint)b0;
                cplen = 1;
            }
            // Map GPT-2 unicode codepoint back to original byte
            int byte_val = gpt2_byte_decode(cp);
            if (byte_val >= 0) {
                buf[out] = (char)byte_val;
                out++;
            }
            i += cplen;
        }
        if (out == 0) return "";
        return (String)buf[0..out - 1];
    }

    // SPM tokens: replace ▁ (U+2581, bytes E2 96 81) with space
    usz out = 0;
    usz i = 0;
    while (i < piece.len) {
        if (i + 2 < piece.len &&
            piece[i] == (char)0xe2 &&
            piece[i + 1] == (char)0x96 &&
            piece[i + 2] == (char)0x81) {
            buf[out] = ' ';
            out++;
            i += 3;
        } else {
            buf[out] = piece[i];
            out++;
            i++;
        }
    }

    if (out == 0) return "";
    return (String)buf[0..out - 1];
}

// --- Encode dispatcher ---

fn usz utf8_codepoint_len(char first_byte) {
    if ((first_byte & 0x80) == 0) return 1;
    if ((first_byte & 0xe0) == 0xc0) return 2;
    if ((first_byte & 0xf0) == 0xe0) return 3;
    if ((first_byte & 0xf8) == 0xf0) return 4;
    return 1;
}

struct BPESymbol {
    usz text_start;
    usz text_len;
    int prev;
    int next;
}

fn uint[]? Tokenizer.encode(&self, String text) {
    if (self.is_bpe) {
        return self.encode_bpe(text);
    }
    return self.encode_spm(text);
}

// --- SPM encode (original SentencePiece BPE) ---

fn uint[]? Tokenizer.encode_spm(&self, String text) {
    if (text.len == 0) return mem::new_array(uint, 0);

    // Build escaped text: prepend ▁, replace spaces with ▁ (3 bytes: E2 96 81)
    usz max_esc = text.len * 3 + 3;
    char[] esc = mem::new_array(char, max_esc);
    defer mem::free(esc);

    usz esc_len = 0;
    esc[0] = (char)0xe2; esc[1] = (char)0x96; esc[2] = (char)0x81;
    esc_len = 3;

    for (usz i = 0; i < text.len; i++) {
        if (text[i] == ' ') {
            esc[esc_len]     = (char)0xe2;
            esc[esc_len + 1] = (char)0x96;
            esc[esc_len + 2] = (char)0x81;
            esc_len += 3;
        } else {
            esc[esc_len] = text[i];
            esc_len++;
        }
    }

    // Split into UTF-8 codepoints -> initial BPE symbols
    usz max_syms = esc_len * 4;
    BPESymbol[] syms = mem::new_array(BPESymbol, max_syms);
    defer mem::free(syms);

    usz n_syms = 0;
    usz pos = 0;
    while (pos < esc_len) {
        usz cplen = utf8_codepoint_len(esc[pos]);
        if (pos + cplen > esc_len) cplen = 1;

        String cp_text = (String)esc[pos .. pos + cplen - 1];
        uint dummy_id;
        float dummy_score;
        if (self.vocab_map.lookup(cp_text, &dummy_id, &dummy_score)) {
            syms[n_syms] = {
                .text_start = pos,
                .text_len = cplen,
                .prev = (int)n_syms - 1,
                .next = (int)n_syms + 1,
            };
            n_syms++;
        } else {
            // Byte fallback: split into individual bytes
            for (usz b = 0; b < cplen; b++) {
                syms[n_syms] = {
                    .text_start = pos + b,
                    .text_len = 1,
                    .prev = (int)n_syms - 1,
                    .next = (int)n_syms + 1,
                };
                n_syms++;
            }
        }
        pos += cplen;
    }

    if (n_syms == 0) return mem::new_array(uint, 0);
    syms[n_syms - 1].next = -1;

    // BPE merge loop: repeatedly merge the highest-scored adjacent pair
    while (true) {
        float best_score = -1e30;
        int best_sym = -1;

        int s = 0;
        while (s != -1) {
            int nx = syms[(usz)s].next;
            if (nx != -1) {
                usz merged_start = syms[(usz)s].text_start;
                usz merged_len = syms[(usz)s].text_len + syms[(usz)nx].text_len;
                String merged = (String)esc[merged_start .. merged_start + merged_len - 1];

                uint id;
                float score;
                if (self.vocab_map.lookup(merged, &id, &score) && score > best_score) {
                    best_score = score;
                    best_sym = s;
                }
            }
            s = syms[(usz)s].next;
        }

        if (best_sym == -1) break;

        // Merge best_sym with its next
        int nx = syms[(usz)best_sym].next;
        syms[(usz)best_sym].text_len += syms[(usz)nx].text_len;
        int nn = syms[(usz)nx].next;
        syms[(usz)best_sym].next = nn;
        if (nn != -1) syms[(usz)nn].prev = best_sym;
    }

    // Count final tokens
    usz n_tokens = 0;
    int s = 0;
    while (s != -1) {
        n_tokens++;
        s = syms[(usz)s].next;
    }

    // Build output token array
    uint[] tokens = mem::new_array(uint, n_tokens);
    s = 0;
    usz ti = 0;
    while (s != -1) {
        usz sym_start = syms[(usz)s].text_start;
        usz sym_len = syms[(usz)s].text_len;
        String piece = (String)esc[sym_start .. sym_start + sym_len - 1];

        uint id;
        float score;
        if (self.vocab_map.lookup(piece, &id, &score)) {
            tokens[ti] = id;
        } else if (sym_len == 1) {
            tokens[ti] = self.byte_tokens[(usz)esc[sym_start]];
        } else {
            tokens[ti] = 0;
        }
        ti++;
        s = syms[(usz)s].next;
    }

    return tokens;
}

// --- BPE encode (for LLaMA 3 / GPT-2 style) ---

// Pre-tokenize: split text into words for BPE processing
// Simplified ASCII-based approach:
//   1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd
//   2. Letter sequences (including UTF-8 multi-byte)
//   3. Digit groups (1-3 digits)
//   4. Punctuation sequences
//   5. Whitespace sequences

fn bool is_utf8_cont(char c) {
    return (c & 0xC0) == 0x80;
}

// Check if text at position matches a contraction suffix (after ')
fn usz match_contraction(String text, usz pos) {
    if (pos >= text.len || text[pos] != '\'') return 0;
    usz rem = text.len - pos;
    // Check 'll, 're, 've
    if (rem >= 3) {
        char c1 = text[pos + 1];
        char c2 = text[pos + 2];
        if (c1 == 'l' && c2 == 'l') return 3;
        if (c1 == 'L' && c2 == 'L') return 3;
        if (c1 == 'r' && c2 == 'e') return 3;
        if (c1 == 'R' && c2 == 'E') return 3;
        if (c1 == 'v' && c2 == 'e') return 3;
        if (c1 == 'V' && c2 == 'E') return 3;
    }
    // Check 's, 't, 'm, 'd
    if (rem >= 2) {
        char c1 = text[pos + 1];
        if (c1 == 's' || c1 == 'S') return 2;
        if (c1 == 't' || c1 == 'T') return 2;
        if (c1 == 'm' || c1 == 'M') return 2;
        if (c1 == 'd' || c1 == 'D') return 2;
    }
    return 0;
}

struct WordSpan {
    usz start;
    usz len;
}

fn WordSpan[] pre_tokenize(String text, usz* out_count) {
    usz max_words = text.len + 1;
    WordSpan[] words = mem::new_array(WordSpan, max_words);
    usz n_words = 0;
    usz pos = 0;

    while (pos < text.len) {
        // Try contraction
        usz clen = match_contraction(text, pos);
        if (clen > 0) {
            words[n_words] = { .start = pos, .len = clen };
            n_words++;
            pos += clen;
            continue;
        }

        char c = text[pos];
        usz start = pos;

        // Optionally consume a single leading space (0x20) before non-whitespace.
        // This matches GPT-2 regex patterns like ` ?\p{L}+` which group space with word.
        if (c == ' ' && pos + 1 < text.len && !ascii::is_space_m(text[pos + 1])) {
            pos++;
            c = text[pos];
        }

        // Letters (including UTF-8 multi-byte sequences)
        if (ascii::is_alpha_m(c) || (c & 0x80) != 0) {
            pos++;
            while (pos < text.len) {
                char nc = text[pos];
                if (ascii::is_alpha_m(nc) || is_utf8_cont(nc) || (nc & 0x80) != 0) {
                    pos++;
                } else {
                    break;
                }
            }
            words[n_words] = { .start = start, .len = pos - start };
            n_words++;
            continue;
        }

        // Digits (1-3 at a time)
        if (ascii::is_digit_m(c)) {
            usz count = 0;
            while (pos < text.len && ascii::is_digit_m(text[pos]) && count < 3) {
                pos++;
                count++;
            }
            words[n_words] = { .start = start, .len = pos - start };
            n_words++;
            continue;
        }

        // Whitespace
        if (ascii::is_space_m(c)) {
            while (pos < text.len && ascii::is_space_m(text[pos])) {
                pos++;
            }
            words[n_words] = { .start = start, .len = pos - start };
            n_words++;
            continue;
        }

        // Punctuation / other (single char, possibly with leading space)
        pos++;
        words[n_words] = { .start = start, .len = pos - start };
        n_words++;
    }

    *out_count = n_words;
    return words;
}

// BPE merge on a single word using rank-based merges
fn void bpe_merge_word(BPESymbol[] syms, usz n_syms, char[] text, MergeMap* merges) {
    if (n_syms <= 1) return;

    while (true) {
        uint best_rank = MERGE_NOT_FOUND;
        int best_sym = -1;

        int s = 0;
        while (s != -1) {
            int nx = syms[(usz)s].next;
            if (nx != -1) {
                String a = (String)text[syms[(usz)s].text_start .. syms[(usz)s].text_start + syms[(usz)s].text_len - 1];
                String b = (String)text[syms[(usz)nx].text_start .. syms[(usz)nx].text_start + syms[(usz)nx].text_len - 1];
                uint rank = merges.lookup(a, b);
                if (rank != MERGE_NOT_FOUND && rank < best_rank) {
                    best_rank = rank;
                    best_sym = s;
                }
            }
            s = syms[(usz)s].next;
        }

        if (best_sym == -1) break;

        // Merge best_sym with its next
        int nx = syms[(usz)best_sym].next;
        syms[(usz)best_sym].text_len += syms[(usz)nx].text_len;
        int nn = syms[(usz)nx].next;
        syms[(usz)best_sym].next = nn;
        if (nn != -1) syms[(usz)nn].prev = best_sym;
    }
}

fn uint[]? Tokenizer.encode_bpe(&self, String text) {
    if (text.len == 0) return mem::new_array(uint, 0);

    // Pre-tokenize into words (on raw text)
    usz n_words;
    WordSpan[] words = pre_tokenize(text, &n_words);
    defer mem::free(words);

    // Collect all tokens into a dynamic list
    usz max_tokens = text.len + n_words;
    uint[] all_tokens = mem::new_array(uint, max_tokens);
    usz total_tokens = 0;

    // Buffer for GPT-2 byte encoding (each raw byte -> up to 2 UTF-8 bytes)
    usz enc_buf_size = text.len * 3 + 16;
    char[] enc_buf = mem::new_array(char, enc_buf_size);
    defer mem::free(enc_buf);

    // Allocate shared BPE symbol buffer
    usz max_syms = text.len * 4;
    BPESymbol[] syms = mem::new_array(BPESymbol, max_syms);
    defer mem::free(syms);

    for (usz w = 0; w < n_words; w++) {
        String word = (String)text[words[w].start .. words[w].start + words[w].len - 1];

        // GPT-2 byte encode the word: raw bytes -> unicode codepoints -> UTF-8
        // The vocab stores GPT-2-encoded strings (e.g. space=Ġ, newline=Ċ)
        usz enc_len = gpt2_encode_to_buf(word, enc_buf);
        String enc_word = (String)enc_buf[0..enc_len - 1];

        // Try ignore_merges: look up whole encoded word in vocab first
        if (self.ignore_merges) {
            uint word_id;
            if (self.vocab_map.lookup_id(enc_word, &word_id)) {
                all_tokens[total_tokens] = word_id;
                total_tokens++;
                continue;
            }
        }

        // Split encoded word into UTF-8 codepoints as initial BPE symbols
        usz n_syms = 0;
        usz pos = 0;
        while (pos < enc_len) {
            usz cplen = utf8_codepoint_len(enc_buf[pos]);
            if (pos + cplen > enc_len) cplen = 1;
            syms[n_syms] = {
                .text_start = pos,
                .text_len = cplen,
                .prev = (int)n_syms - 1,
                .next = (int)n_syms + 1,
            };
            n_syms++;
            pos += cplen;
        }

        if (n_syms == 0) continue;
        syms[n_syms - 1].next = -1;

        // Run BPE merges on encoded form
        bpe_merge_word(syms, n_syms, enc_buf, &self.merge_map);

        // Collect resulting tokens
        int s = 0;
        while (s != -1) {
            usz sym_start = syms[(usz)s].text_start;
            usz sym_len = syms[(usz)s].text_len;
            String piece = (String)enc_buf[sym_start .. sym_start + sym_len - 1];

            uint id;
            if (self.vocab_map.lookup_id(piece, &id)) {
                all_tokens[total_tokens] = id;
            } else if (sym_len == 1) {
                // Single encoded byte not in vocab — try byte token fallback
                all_tokens[total_tokens] = self.byte_tokens[(usz)enc_buf[sym_start]];
            } else {
                // Multi-byte unknown: fall back to individual codepoints
                for (usz b = 0; b < sym_len; b++) {
                    all_tokens[total_tokens] = self.byte_tokens[(usz)enc_buf[sym_start + b]];
                    total_tokens++;
                }
                s = syms[(usz)s].next;
                continue;
            }
            total_tokens++;
            s = syms[(usz)s].next;
        }
    }

    // Trim to actual size
    uint[] result = mem::new_array(uint, total_tokens);
    for (usz i = 0; i < total_tokens; i++) {
        result[i] = all_tokens[i];
    }
    mem::free(all_tokens);

    return result;
}

// Find a special/control token by its string representation.
// Scans the full vocabulary (O(n)) — call sparingly, cache results.
fn uint Tokenizer.find_special_token(&self, String text) {
    for (usz i = 0; i < self.vocab_size; i++) {
        if (self.token_types[i] == TOKEN_CONTROL && self.vocab[i] == text) {
            return (uint)i;
        }
    }
    return 0;
}

// Encode text containing special token patterns (e.g. <|im_start|>, <|im_end|>).
// Splits input around special tokens, BPE-encodes regular segments, inserts special token IDs.
fn uint[]? Tokenizer.encode_with_specials(&self, String text) {
    if (text.len == 0) return mem::new_array(uint, 0);

    // Find special token IDs (cached per call — fast enough for one-time use)
    uint im_start_id = self.find_special_token("<|im_start|>");
    uint im_end_id = self.find_special_token("<|im_end|>");

    // Max possible output tokens
    uint[] result = mem::new_array(uint, text.len + 32);
    usz n = 0;
    usz pos = 0;

    while (pos < text.len) {
        usz remaining = text.len - pos;

        // Check for <|im_start|> (12 chars)
        if (remaining >= 12 && im_start_id != 0) {
            String candidate = (String)text[pos..pos + 11];
            if (candidate == "<|im_start|>") {
                result[n] = im_start_id;
                n++;
                pos += 12;
                continue;
            }
        }

        // Check for <|im_end|> (10 chars)
        if (remaining >= 10 && im_end_id != 0) {
            String candidate = (String)text[pos..pos + 9];
            if (candidate == "<|im_end|>") {
                result[n] = im_end_id;
                n++;
                pos += 10;
                continue;
            }
        }

        // Find end of regular text segment (next special token or end)
        usz seg_end = pos + 1;
        while (seg_end < text.len) {
            usz rem = text.len - seg_end;
            if (rem >= 12 && im_start_id != 0) {
                String c2 = (String)text[seg_end..seg_end + 11];
                if (c2 == "<|im_start|>") break;
            }
            if (rem >= 10 && im_end_id != 0) {
                String c2 = (String)text[seg_end..seg_end + 9];
                if (c2 == "<|im_end|>") break;
            }
            seg_end++;
        }

        // Tokenize the regular segment
        String segment = (String)text[pos..seg_end - 1];
        uint[] seg_tokens = self.encode(segment)!;
        for (usz i = 0; i < seg_tokens.len; i++) {
            result[n] = seg_tokens[i];
            n++;
        }
        mem::free(seg_tokens);
        pos = seg_end;
    }

    // Trim to actual size
    uint[] trimmed = mem::new_array(uint, n);
    for (usz i = 0; i < n; i++) trimmed[i] = result[i];
    mem::free(result);
    return trimmed;
}

fn void Tokenizer.free(&self) {
    if (self.vocab.len > 0) mem::free(self.vocab);
    if (self.token_types.len > 0) mem::free(self.token_types);
    if (self.scores.len > 0) mem::free(self.scores);
    self.vocab_map.free_map();
    if (self.is_bpe) {
        self.merge_map.free_map();
    }
}
