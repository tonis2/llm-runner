module llm;

import vk;
import std::io;
import std::core::mem;

// LoRA (Low-Rank Adaptation) loader and runtime data structures.
//
// LoRA augments a base linear layer  y = W x  with a low-rank correction:
//
//     y = W x + (alpha / rank) * B @ A @ x
//
// where A has shape [rank, in_dim] and B has shape [out_dim, rank]. We
// always upload A and B as F32 (the tensors are tiny) so the existing F32
// batch_matmul kernels can be reused for hop 1 (A) and the new
// batch_matmul_add_scaled kernel can do hop 2 (B + scaled accumulate).

// One adapted linear: A and B uploaded to GPU as F32, plus the precomputed
// scale = strength * alpha / rank. `present` is false when the LoRA file
// did not contain a pair for this site (it falls through to the base call
// untouched at compute time).
struct LoraTensor {
    Tensor a;        // [rank, in_dim] F32
    Tensor b;        // [out_dim, rank] F32
    uint   rank;
    float  scale;
    bool   present;
}

fn void LoraTensor.free(&self) {
    if (self.present) {
        self.a.free();
        self.b.free();
        self.present = false;
    }
}

// LoRA pairs for one DiT main-block layer. Q/K/V are tracked separately
// even though the base wqkv is fused — each base call writes to a distinct
// destination buffer (a.q / a.k / a.v) so the LoRA correction is per-slot.
struct DiTLoraLayer {
    LoraTensor wq;
    LoraTensor wk;
    LoraTensor wv;
    LoraTensor wo;
    LoraTensor ffn_gate; // w1
    LoraTensor ffn_up;   // w3
    LoraTensor ffn_down; // w2
}

fn void DiTLoraLayer.free(&self) {
    self.wq.free();
    self.wk.free();
    self.wv.free();
    self.wo.free();
    self.ffn_gate.free();
    self.ffn_up.free();
    self.ffn_down.free();
}

// Top-level adapter: a heap-allocated array of per-layer bundles plus the
// scalars used at load time. Layers that have no LoRA pairs are still in
// the array but their LoraTensor.present flags are all false.
struct DiTLoraAdapter {
    String path;
    float  strength;        // user-supplied weight (0.0 disables effect)
    float  alpha;           // from __metadata__, defaults to rank if absent
    DiTLoraLayer* layers;   // heap-allocated, length = n_layers
    uint   n_layers;
    uint   covered_sites;   // total LoraTensors with present == true
    bool   loaded;
}

fn void DiTLoraAdapter.free(&self) {
    if (!self.loaded) return;
    for (uint l = 0; l < self.n_layers; l++) {
        self.layers[l].free();
    }
    mem::free(self.layers);
    self.loaded = false;
}

// --- Naming schemes ---
//
// Different LoRA trainers emit different tensor name layouts. We pick a
// scheme at the first layer that yields any matches and lock it in for the
// rest of the load. Each scheme provides a layer prefix template and the
// per-module suffixes for Q/K/V/O and the FFN gate/up/down. The suffix is
// the part *before* `.lora_A.weight` / `.lora_B.weight` (or
// `.lora_down.weight` / `.lora_up.weight` — Kohya's alternate convention).

struct LoraNameScheme {
    String layer_prefix;   // appended with the layer index, e.g. "transformer_blocks."
    String layer_suffix;   // appended after the layer index, e.g. "."
    String wq;
    String wk;
    String wv;
    String wo;
    String ffn_gate;
    String ffn_up;
    String ffn_down;
}

const LoraNameScheme[*] LORA_SCHEMES = {
    // 1. Diffusers Lumina2 / native Z-Image attention.to_{q,k,v,out}
    {
        .layer_prefix = "transformer_blocks.",
        .layer_suffix = ".",
        .wq           = "attention.to_q",
        .wk           = "attention.to_k",
        .wv           = "attention.to_v",
        .wo           = "attention.to_out",
        .ffn_gate     = "feed_forward.w1",
        .ffn_up       = "feed_forward.w3",
        .ffn_down     = "feed_forward.w2",
    },
    // 2. PEFT-wrapped Diffusers (with leading "transformer." container)
    {
        .layer_prefix = "transformer.transformer_blocks.",
        .layer_suffix = ".",
        .wq           = "attention.to_q",
        .wk           = "attention.to_k",
        .wv           = "attention.to_v",
        .wo           = "attention.to_out",
        .ffn_gate     = "feed_forward.w1",
        .ffn_up       = "feed_forward.w3",
        .ffn_down     = "feed_forward.w2",
    },
    // 3. PEFT-wrapped with base_model.model. prefix
    {
        .layer_prefix = "base_model.model.transformer.transformer_blocks.",
        .layer_suffix = ".",
        .wq           = "attention.to_q",
        .wk           = "attention.to_k",
        .wv           = "attention.to_v",
        .wo           = "attention.to_out",
        .ffn_gate     = "feed_forward.w1",
        .ffn_up       = "feed_forward.w3",
        .ffn_down     = "feed_forward.w2",
    },
    // 4. ComfyUI / Kohya-converted "diffusion_model." prefix
    {
        .layer_prefix = "diffusion_model.layers.",
        .layer_suffix = ".",
        .wq           = "attention.to_q",
        .wk           = "attention.to_k",
        .wv           = "attention.to_v",
        .wo           = "attention.to_out",
        .ffn_gate     = "feed_forward.w1",
        .ffn_up       = "feed_forward.w3",
        .ffn_down     = "feed_forward.w2",
    },
};

// Append a String to a char buffer at position `pos`, returning the new position.
fn usz append_str(char[256]* buf, usz pos, String s) {
    for (usz i = 0; i < s.len; i++) {
        if (pos >= 256) break;
        (*buf)[pos] = s[i];
        pos++;
    }
    return pos;
}

// Append decimal representation of a uint at `pos`.
fn usz append_uint(char[256]* buf, usz pos, uint v) {
    if (v >= 100) { (*buf)[pos] = (char)('0' + v / 100); pos++; }
    if (v >= 10)  { (*buf)[pos] = (char)('0' + (v / 10) % 10); pos++; }
    (*buf)[pos] = (char)('0' + v % 10); pos++;
    return pos;
}

// Build "{prefix}{layer}{suffix}{module}" into the buffer; return its String slice.
fn String build_module_path(char[256]* buf, LoraNameScheme* scheme, uint layer, String module_suffix) {
    usz pos = 0;
    pos = append_str(buf, pos, scheme.layer_prefix);
    pos = append_uint(buf, pos, layer);
    pos = append_str(buf, pos, scheme.layer_suffix);
    pos = append_str(buf, pos, module_suffix);
    return (String)(*buf)[0:pos];
}

// One pass over a candidate (prefix, module, AB-suffix) tuple. Try the
// `lora_A` / `lora_B` pair first, then fall back to Kohya's `lora_down` /
// `lora_up`. Returns the matched A and B tensor names, or empty strings if
// neither pair exists.
fn bool find_lora_pair_names(
    SafetensorsFile* sf,
    String full_module,
    char[256]* a_buf,
    char[256]* b_buf,
    String* a_out,
    String* b_out,
) {
    // Try .lora_A.weight / .lora_B.weight (Diffusers / PEFT)
    usz pa = append_str(a_buf, 0, full_module);
    pa = append_str(a_buf, pa, ".lora_A.weight");
    usz pb = append_str(b_buf, 0, full_module);
    pb = append_str(b_buf, pb, ".lora_B.weight");
    String a_name = (String)(*a_buf)[0:pa];
    String b_name = (String)(*b_buf)[0:pb];
    if (try a_info = sf.find_tensor(a_name)) {
        if (try b_info = sf.find_tensor(b_name)) {
            *a_out = a_name;
            *b_out = b_name;
            return true;
        }
    }
    // Try .lora_down.weight / .lora_up.weight (Kohya)
    pa = append_str(a_buf, 0, full_module);
    pa = append_str(a_buf, pa, ".lora_down.weight");
    pb = append_str(b_buf, 0, full_module);
    pb = append_str(b_buf, pb, ".lora_up.weight");
    a_name = (String)(*a_buf)[0:pa];
    b_name = (String)(*b_buf)[0:pb];
    if (try a_info2 = sf.find_tensor(a_name)) {
        if (try b_info2 = sf.find_tensor(b_name)) {
            *a_out = a_name;
            *b_out = b_name;
            return true;
        }
    }
    return false;
}

// Optional per-tensor alpha override. Some LoRA files ship a small `<module>.alpha`
// scalar tensor that overrides the global lora_alpha for that specific pair.
// Returns the override if present, or -1.0f if absent.
fn float try_load_per_tensor_alpha(SafetensorsFile* sf, String full_module) {
    char[256] buf;
    usz pos = append_str(&buf, 0, full_module);
    pos = append_str(&buf, pos, ".alpha");
    String name = (String)buf[0:pos];
    if (try info = sf.find_tensor(name)) {
        // Should be a scalar — read first F32 element directly.
        if (info.dtype == "F32") {
            char* data_ptr = sf.data_base + (usz)info.data_start;
            return *(float*)data_ptr;
        }
        if (info.dtype == "F16") {
            char* data_ptr = sf.data_base + (usz)info.data_start;
            return cpu_f16_to_f32(data_ptr);
        }
        if (info.dtype == "BF16") {
            char* data_ptr = sf.data_base + (usz)info.data_start;
            return cpu_bf16_to_f32(data_ptr);
        }
    }
    return -1.0f;
}

// Try to load one LoraTensor for a (layer, module-suffix) site under the
// given scheme. Returns a populated LoraTensor with present=true on success,
// or a zero-initialised one with present=false on failure (so the caller
// can store it directly).
fn LoraTensor try_load_lora_site(
    DeviceContext* ctx,
    SafetensorsFile* sf,
    LoraNameScheme* scheme,
    uint layer,
    String module_suffix,
    float global_alpha,
    float strength,
) {
    LoraTensor result;
    char[256] mod_buf;
    String full_module = build_module_path(&mod_buf, scheme, layer, module_suffix);

    char[256] a_buf;
    char[256] b_buf;
    String a_name;
    String b_name;
    if (!find_lora_pair_names(sf, full_module, &a_buf, &b_buf, &a_name, &b_name)) {
        return result;
    }

    // Upload both halves as F32 (handles F16/BF16/F32/FP8 source).
    Tensor a_tensor;
    Tensor b_tensor;
    if (try t = upload_safetensor_f32(ctx, sf, a_name)) {
        a_tensor = t;
    } else {
        return result;
    }
    if (try t = upload_safetensor_f32(ctx, sf, b_name)) {
        b_tensor = t;
    } else {
        a_tensor.free();
        return result;
    }

    // Rank = first dim of A (after the [col, row] reversal in upload_safetensor_f32,
    // shape[0] is the contraction axis = in_dim, shape[1] is rank).
    uint rank = (uint)a_tensor.shape[1];
    if (rank == 0) {
        a_tensor.free();
        b_tensor.free();
        return result;
    }

    // Per-tensor alpha overrides global if present.
    float effective_alpha = try_load_per_tensor_alpha(sf, full_module);
    if (effective_alpha < 0.0f) effective_alpha = global_alpha;
    if (effective_alpha <= 0.0f) effective_alpha = (float)rank; // safe default

    result.a = a_tensor;
    result.b = b_tensor;
    result.rank = rank;
    result.scale = strength * effective_alpha / (float)rank;
    result.present = true;
    return result;
}

// Pick the first scheme that yields any matches at layer 0.
fn LoraNameScheme* detect_lora_scheme(SafetensorsFile* sf) {
    String[7] modules = { "attention.to_q", "attention.to_k", "attention.to_v",
                          "attention.to_out", "feed_forward.w1",
                          "feed_forward.w3", "feed_forward.w2" };
    for (uint i = 0; i < LORA_SCHEMES.len; i++) {
        LoraNameScheme* s = &LORA_SCHEMES[i];
        for (uint m = 0; m < 7; m++) {
            char[256] mod_buf;
            String full = build_module_path(&mod_buf, s, 0, modules[m]);

            char[256] a_buf;
            usz pa = append_str(&a_buf, 0, full);
            pa = append_str(&a_buf, pa, ".lora_A.weight");
            String a_name = (String)a_buf[0:pa];
            if (try _info = sf.find_tensor(a_name)) return s;

            pa = append_str(&a_buf, 0, full);
            pa = append_str(&a_buf, pa, ".lora_down.weight");
            a_name = (String)a_buf[0:pa];
            if (try _info = sf.find_tensor(a_name)) return s;
        }
    }
    return null;
}

// Public entry point: open a safetensors file, autodetect the naming
// scheme, and populate a DiTLoraAdapter for `n_layers` main DiT blocks.
fn DiTLoraAdapter? load_dit_lora_safetensors(
    DeviceContext* ctx,
    String path,
    float strength,
    uint n_layers,
) {
    io::printfn("  Loading LoRA: %s", path);
    SafetensorsFile sf = safetensors_open(path)!;
    defer sf.close();

    // Read alpha from __metadata__: try a few common keys.
    float alpha = 0.0f;
    if (try v = sf.get_metadata("lora_alpha")) {
        alpha = (float)(v.to_double() ?? 0.0);
    } else if (try v = sf.get_metadata("ss_network_alpha")) {
        alpha = (float)(v.to_double() ?? 0.0);
    } else if (try v = sf.get_metadata("alpha")) {
        alpha = (float)(v.to_double() ?? 0.0);
    }
    if (alpha == 0.0f) {
        io::printfn("  LoRA: no alpha in __metadata__, will default to rank per pair");
    }

    LoraNameScheme* scheme = detect_lora_scheme(&sf);
    if (scheme == null) {
        io::printfn("  LoRA: no recognised tensor naming scheme. Header keys (first 8):");
        usz max = sf.n_tensors < 8 ? sf.n_tensors : 8;
        for (usz i = 0; i < max; i++) {
            io::printfn("    %s", sf.tensors[i].name);
        }
        return COMPUTE_ERROR~;
    }
    io::printfn("  LoRA scheme: prefix=\"%s\"", scheme.layer_prefix);

    DiTLoraLayer[] layers = mem::new_array(DiTLoraLayer, n_layers);

    uint covered = 0;
    for (uint l = 0; l < n_layers; l++) {
        layers[l].wq        = try_load_lora_site(ctx, &sf, scheme, l, "attention.to_q",   alpha, strength);
        layers[l].wk        = try_load_lora_site(ctx, &sf, scheme, l, "attention.to_k",   alpha, strength);
        layers[l].wv        = try_load_lora_site(ctx, &sf, scheme, l, "attention.to_v",   alpha, strength);
        layers[l].wo        = try_load_lora_site(ctx, &sf, scheme, l, "attention.to_out", alpha, strength);
        layers[l].ffn_gate  = try_load_lora_site(ctx, &sf, scheme, l, "feed_forward.w1",  alpha, strength);
        layers[l].ffn_up    = try_load_lora_site(ctx, &sf, scheme, l, "feed_forward.w3",  alpha, strength);
        layers[l].ffn_down  = try_load_lora_site(ctx, &sf, scheme, l, "feed_forward.w2",  alpha, strength);

        if (layers[l].wq.present)       covered++;
        if (layers[l].wk.present)       covered++;
        if (layers[l].wv.present)       covered++;
        if (layers[l].wo.present)       covered++;
        if (layers[l].ffn_gate.present) covered++;
        if (layers[l].ffn_up.present)   covered++;
        if (layers[l].ffn_down.present) covered++;
    }

    uint total_sites = n_layers * 7;
    io::printfn("  LoRA loaded: alpha=%.2f strength=%.2f, covered %d/%d sites",
        alpha, strength, covered, total_sites);

    return {
        .path = path,
        .strength = strength,
        .alpha = alpha,
        .layers = &layers[0],
        .n_layers = n_layers,
        .covered_sites = covered,
        .loaded = true,
    };
}
