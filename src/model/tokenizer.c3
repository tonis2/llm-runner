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

const usz MERGE_MAP_CAPACITY = 262144;
const uint MERGE_NOT_FOUND = 0xFFFFFFFF;

struct MergeMap {
    String[] keys_a;
    String[] keys_b;
    uint[] ranks;
    bool[] occupied;
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

fn MergeMap create_merge_map() {
    MergeMap map;
    map.keys_a = mem::new_array(String, MERGE_MAP_CAPACITY);
    map.keys_b = mem::new_array(String, MERGE_MAP_CAPACITY);
    map.ranks = mem::new_array(uint, MERGE_MAP_CAPACITY);
    map.occupied = mem::new_array(bool, MERGE_MAP_CAPACITY);
    for (usz i = 0; i < MERGE_MAP_CAPACITY; i++) {
        map.occupied[i] = false;
    }
    return map;
}

fn void MergeMap.insert(&self, String a, String b, uint rank) {
    usz idx = merge_hash(a, b) & (MERGE_MAP_CAPACITY - 1);
    while (self.occupied[idx]) {
        idx = (idx + 1) & (MERGE_MAP_CAPACITY - 1);
    }
    self.keys_a[idx] = a;
    self.keys_b[idx] = b;
    self.ranks[idx] = rank;
    self.occupied[idx] = true;
}

fn uint MergeMap.lookup(&self, String a, String b) {
    usz idx = merge_hash(a, b) & (MERGE_MAP_CAPACITY - 1);
    usz start = idx;
    while (self.occupied[idx]) {
        if (str_eq(self.keys_a[idx], a) && str_eq(self.keys_b[idx], b)) {
            return self.ranks[idx];
        }
        idx = (idx + 1) & (MERGE_MAP_CAPACITY - 1);
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
    bool is_bpe = str_eq(tok_model, "gpt2");

    // Read scores (may not exist for BPE models)
    float[] scores;
    float[]? scores_result = gf.read_f32_array("tokenizer.ggml.scores");
    if (try scores_result) {
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
        merge_map = create_merge_map();
        String[]? merges_result = gf.read_string_array("tokenizer.ggml.merges");
        if (try merges_result) {
            String[] merges = merges_result;
            for (usz i = 0; i < merges.len; i++) {
                // Each merge line is "token_a token_b" separated by space
                String line = merges[i];
                usz split = 0;
                for (usz j = 0; j < line.len; j++) {
                    if (line[j] == ' ') { split = j; break; }
                }
                if (split > 0 && split < line.len - 1) {
                    String a = (String)line[0..split - 1];
                    String b = (String)line[split + 1..line.len - 1];
                    merge_map.insert(a, b, (uint)i);
                }
            }
            io::printfn("  Loaded %d BPE merges", merges.len);
            mem::free(merges);
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

    // BPE tokens are literal — no ▁ replacement needed
    if (self.is_bpe) {
        return piece;
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

fn bool is_letter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

fn bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

fn bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

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

        // Letters (including UTF-8 multi-byte sequences)
        if (is_letter(c) || (c & 0x80) != 0) {
            usz start = pos;
            pos++;
            while (pos < text.len) {
                char nc = text[pos];
                if (is_letter(nc) || is_utf8_cont(nc) || (nc & 0x80) != 0) {
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
        if (is_digit(c)) {
            usz start = pos;
            usz count = 0;
            while (pos < text.len && is_digit(text[pos]) && count < 3) {
                pos++;
                count++;
            }
            words[n_words] = { .start = start, .len = pos - start };
            n_words++;
            continue;
        }

        // Whitespace
        if (is_space(c)) {
            usz start = pos;
            while (pos < text.len && is_space(text[pos])) {
                pos++;
            }
            words[n_words] = { .start = start, .len = pos - start };
            n_words++;
            continue;
        }

        // Punctuation / other (single char)
        words[n_words] = { .start = pos, .len = 1 };
        n_words++;
        pos++;
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

    // Pre-tokenize into words
    usz n_words;
    WordSpan[] words = pre_tokenize(text, &n_words);
    defer mem::free(words);

    // Collect all tokens into a dynamic list
    usz max_tokens = text.len + n_words;
    uint[] all_tokens = mem::new_array(uint, max_tokens);
    usz total_tokens = 0;

    // Allocate shared BPE symbol buffer
    usz max_syms = text.len * 2;
    BPESymbol[] syms = mem::new_array(BPESymbol, max_syms);
    defer mem::free(syms);

    for (usz w = 0; w < n_words; w++) {
        String word = (String)text[words[w].start .. words[w].start + words[w].len - 1];

        // Try ignore_merges: look up whole word in vocab first
        if (self.ignore_merges) {
            uint word_id;
            if (self.vocab_map.lookup_id(word, &word_id)) {
                all_tokens[total_tokens] = word_id;
                total_tokens++;
                continue;
            }
        }

        // Split word into UTF-8 codepoints as initial symbols
        usz n_syms = 0;
        usz pos = 0;
        while (pos < word.len) {
            usz cplen = utf8_codepoint_len(word[pos]);
            if (pos + cplen > word.len) cplen = 1;
            syms[n_syms] = {
                .text_start = words[w].start + pos,
                .text_len = cplen,
                .prev = (int)n_syms - 1,
                .next = (int)n_syms + 1,
            };
            n_syms++;
            pos += cplen;
        }

        if (n_syms == 0) continue;
        syms[n_syms - 1].next = -1;

        // Run BPE merges
        bpe_merge_word(syms, n_syms, text, &self.merge_map);

        // Collect resulting tokens
        int s = 0;
        while (s != -1) {
            usz sym_start = syms[(usz)s].text_start;
            usz sym_len = syms[(usz)s].text_len;
            String piece = (String)text[sym_start .. sym_start + sym_len - 1];

            uint id;
            if (self.vocab_map.lookup_id(piece, &id)) {
                all_tokens[total_tokens] = id;
            } else if (sym_len == 1) {
                all_tokens[total_tokens] = self.byte_tokens[(usz)text[sym_start]];
            } else {
                // Multi-byte unknown: fall back to individual bytes
                for (usz b = 0; b < sym_len; b++) {
                    all_tokens[total_tokens] = self.byte_tokens[(usz)text[sym_start + b]];
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

fn void Tokenizer.free(&self) {
    if (self.vocab.len > 0) mem::free(self.vocab);
    if (self.token_types.len > 0) mem::free(self.token_types);
    if (self.scores.len > 0) mem::free(self.scores);
    self.vocab_map.free_map();
    if (self.is_bpe) {
        self.merge_map.free_map();
    }
}
