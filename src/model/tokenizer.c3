module llm;

import std::io;
import std::core::mem;

const uint TOKEN_NORMAL  = 1;
const uint TOKEN_CONTROL = 3;
const uint TOKEN_BYTE    = 6;

// --- Vocab hash map (string -> id + score) ---

const usz VOCAB_MAP_CAPACITY = 65536;

struct VocabMap {
    String[] keys;
    uint[] ids;
    float[] scores;
    bool[] occupied;
}

fn usz str_hash(String s) {
    usz h = 14695981039346656037;
    for (usz i = 0; i < s.len; i++) {
        h ^= (usz)s[i];
        h *= 1099511628211;
    }
    return h;
}

fn VocabMap create_vocab_map(String[] vocab, float[] scores, uint[] token_types) {
    VocabMap map;
    map.keys = mem::new_array(String, VOCAB_MAP_CAPACITY);
    map.ids = mem::new_array(uint, VOCAB_MAP_CAPACITY);
    map.scores = mem::new_array(float, VOCAB_MAP_CAPACITY);
    map.occupied = mem::new_array(bool, VOCAB_MAP_CAPACITY);

    for (usz i = 0; i < VOCAB_MAP_CAPACITY; i++) {
        map.occupied[i] = false;
    }

    for (usz i = 0; i < vocab.len; i++) {
        // Only index NORMAL tokens for BPE lookups
        if (token_types[i] == TOKEN_CONTROL || token_types[i] == TOKEN_BYTE) continue;
        usz idx = str_hash(vocab[i]) & (VOCAB_MAP_CAPACITY - 1);
        while (map.occupied[idx]) {
            idx = (idx + 1) & (VOCAB_MAP_CAPACITY - 1);
        }
        map.keys[idx] = vocab[i];
        map.ids[idx] = (uint)i;
        map.scores[idx] = scores[i];
        map.occupied[idx] = true;
    }

    return map;
}

fn bool VocabMap.lookup(&self, String key, uint* out_id, float* out_score) {
    usz idx = str_hash(key) & (VOCAB_MAP_CAPACITY - 1);
    usz start = idx;
    while (self.occupied[idx]) {
        if (str_eq(self.keys[idx], key)) {
            *out_id = self.ids[idx];
            *out_score = self.scores[idx];
            return true;
        }
        idx = (idx + 1) & (VOCAB_MAP_CAPACITY - 1);
        if (idx == start) break;
    }
    return false;
}

fn void VocabMap.free_map(&self) {
    if (self.keys.len > 0) mem::free(self.keys);
    if (self.ids.len > 0) mem::free(self.ids);
    if (self.scores.len > 0) mem::free(self.scores);
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
    float[] scores = gf.read_f32_array("tokenizer.ggml.scores")!;

    uint vocab_size = (uint)vocab.len;
    io::printfn("Tokenizer loaded: %d tokens", vocab_size);

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

    return {
        .vocab = vocab,
        .token_types = token_types,
        .scores = scores,
        .vocab_size = vocab_size,
        .vocab_map = vocab_map,
        .byte_tokens = byte_tokens,
    };
}

// --- Decode (token ID -> text) ---

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

    // Normal tokens: replace ▁ (U+2581, bytes E2 96 81) with space
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

// --- Encode (text -> token IDs) via SentencePiece BPE ---

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
    // Each codepoint not in vocab becomes individual byte symbols (byte fallback)
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
    // syms[0].prev is already -1 from (int)0 - 1

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
            // Byte fallback token
            tokens[ti] = self.byte_tokens[(usz)esc[sym_start]];
        } else {
            tokens[ti] = 0;  // Unknown
        }
        ti++;
        s = syms[(usz)s].next;
    }

    return tokens;
}

fn void Tokenizer.free(&self) {
    if (self.vocab.len > 0) mem::free(self.vocab);
    if (self.token_types.len > 0) mem::free(self.token_types);
    if (self.scores.len > 0) mem::free(self.scores);
    self.vocab_map.free_map();
}
