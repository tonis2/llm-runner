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

// ============================================================================
// Flux 2 Klein DiT LoRA support.
//
// Klein has two block topologies that the Z-Image DiTLoraLayer can't model:
//   - dual-stream blocks: image and text streams each have their own fused
//     QKV [3*dim, dim], output proj [dim, dim], SwiGLU FFN up [2*mlp, dim]
//     and down [dim, mlp].
//   - single-stream blocks: one fused linear1 [3*dim + 2*mlp, dim] (Q/K/V +
//     SwiGLU gate/up) and one fused linear2 [dim, dim+mlp].
//
// LoRA files in the wild use either "native fused" naming (one LoRA pair per
// fused weight, matching the GGUF tensor names) or Diffusers naming
// (separate to_q / to_k / to_v that target sub-ranges of the fused output
// buffers). The runtime applies separate LoraTensors at different row offsets
// of the same destination, so both layouts compose with the existing
// dispatch_lora_correction call.
// ============================================================================

struct FluxDualLoraLayer {
    LoraTensor img_qkv;       // [3*dim, dim] fused
    LoraTensor img_q;         // sub-range [0, dim)
    LoraTensor img_k;         // sub-range [dim, 2*dim)
    LoraTensor img_v;         // sub-range [2*dim, 3*dim)
    LoraTensor img_proj;
    LoraTensor img_mlp_up;    // [2*mlp, dim]
    LoraTensor img_mlp_down;  // [dim, mlp]
    LoraTensor txt_qkv;
    LoraTensor txt_q;
    LoraTensor txt_k;
    LoraTensor txt_v;
    LoraTensor txt_proj;
    LoraTensor txt_mlp_up;
    LoraTensor txt_mlp_down;
}

fn void FluxDualLoraLayer.free(&self) {
    self.img_qkv.free(); self.img_q.free(); self.img_k.free(); self.img_v.free();
    self.img_proj.free(); self.img_mlp_up.free(); self.img_mlp_down.free();
    self.txt_qkv.free(); self.txt_q.free(); self.txt_k.free(); self.txt_v.free();
    self.txt_proj.free(); self.txt_mlp_up.free(); self.txt_mlp_down.free();
}

struct FluxSingleLoraLayer {
    LoraTensor linear1;       // [3*dim + 2*mlp, dim] fused
    LoraTensor q;             // sub-range [0, dim)
    LoraTensor k;             // sub-range [dim, 2*dim)
    LoraTensor v;             // sub-range [2*dim, 3*dim)
    LoraTensor linear2;       // [dim, dim+mlp]
}

fn void FluxSingleLoraLayer.free(&self) {
    self.linear1.free(); self.q.free(); self.k.free(); self.v.free();
    self.linear2.free();
}

struct FluxLoraAdapter {
    String path;
    float  strength;
    float  alpha;
    FluxDualLoraLayer*   dual_layers;
    FluxSingleLoraLayer* single_layers;
    uint   n_dual;
    uint   n_single;
    uint   covered_sites;
    bool   loaded;
}

fn void FluxLoraAdapter.free(&self) {
    if (!self.loaded) return;
    for (uint l = 0; l < self.n_dual; l++)   self.dual_layers[l].free();
    for (uint l = 0; l < self.n_single; l++) self.single_layers[l].free();
    if (self.n_dual > 0)   mem::free(self.dual_layers);
    if (self.n_single > 0) mem::free(self.single_layers);
    self.loaded = false;
}

// Per-scheme module-name suffixes for every Flux LoRA site we know how to
// apply. An empty String means the scheme doesn't expose that site (e.g.
// Diffusers names Q/K/V separately and has no fused QKV name; native-fused
// names QKV as one and has no separate Q/K/V).
struct FluxLoraScheme {
    String dual_prefix;
    String single_prefix;
    String layer_suffix;
    // Dual block sites
    String img_qkv;
    String img_q;
    String img_k;
    String img_v;
    String img_proj;
    String img_mlp_up;
    String img_mlp_down;
    String txt_qkv;
    String txt_q;
    String txt_k;
    String txt_v;
    String txt_proj;
    String txt_mlp_up;
    String txt_mlp_down;
    // Single block sites
    String single_linear1;
    String single_q;
    String single_k;
    String single_v;
    String single_linear2;
}

const FluxLoraScheme[*] FLUX_LORA_SCHEMES = {
    // 1. Native fused — Kohya/ComfyUI/HF flux GGUF-aligned naming.
    {
        .dual_prefix    = "double_blocks.",
        .single_prefix  = "single_blocks.",
        .layer_suffix   = ".",
        .img_qkv        = "img_attn.qkv",
        .img_proj       = "img_attn.proj",
        .img_mlp_up     = "img_mlp.0",
        .img_mlp_down   = "img_mlp.2",
        .txt_qkv        = "txt_attn.qkv",
        .txt_proj       = "txt_attn.proj",
        .txt_mlp_up     = "txt_mlp.0",
        .txt_mlp_down   = "txt_mlp.2",
        .single_linear1 = "linear1",
        .single_linear2 = "linear2",
    },
    // 2. ComfyUI checkpoint export with "diffusion_model." container prefix.
    {
        .dual_prefix    = "diffusion_model.double_blocks.",
        .single_prefix  = "diffusion_model.single_blocks.",
        .layer_suffix   = ".",
        .img_qkv        = "img_attn.qkv",
        .img_proj       = "img_attn.proj",
        .img_mlp_up     = "img_mlp.0",
        .img_mlp_down   = "img_mlp.2",
        .txt_qkv        = "txt_attn.qkv",
        .txt_proj       = "txt_attn.proj",
        .txt_mlp_up     = "txt_mlp.0",
        .txt_mlp_down   = "txt_mlp.2",
        .single_linear1 = "linear1",
        .single_linear2 = "linear2",
    },
    // 3. Diffusers FluxTransformer2DModel naming. Q/K/V are unfused and land
    //    at row sub-ranges of the fused QKV output. MLP sites are deliberately
    //    omitted: Flux1's GELU FFN (single up + single down) doesn't map onto
    //    Klein's SwiGLU FFN (gate/up halves of [2*mlp, dim] + single down), so
    //    a `ff.net.0.proj` LoRA has no correct destination here.
    {
        .dual_prefix    = "transformer.transformer_blocks.",
        .single_prefix  = "transformer.single_transformer_blocks.",
        .layer_suffix   = ".",
        .img_q          = "attn.to_q",
        .img_k          = "attn.to_k",
        .img_v          = "attn.to_v",
        .img_proj       = "attn.to_out.0",
        .txt_q          = "attn.add_q_proj",
        .txt_k          = "attn.add_k_proj",
        .txt_v          = "attn.add_v_proj",
        .txt_proj       = "attn.to_add_out",
        .single_q       = "attn.to_q",
        .single_k       = "attn.to_k",
        .single_v       = "attn.to_v",
        .single_linear2 = "proj_out",
    },
};

// Build "{prefix}{layer}{suffix}{module}" using the same buffer scheme as
// build_module_path but with arbitrary prefix/suffix strings.
fn String flux_build_module_path(char[256]* buf, String layer_prefix, uint layer, String layer_suffix, String module_suffix) {
    usz pos = 0;
    pos = append_str(buf, pos, layer_prefix);
    pos = append_uint(buf, pos, layer);
    pos = append_str(buf, pos, layer_suffix);
    pos = append_str(buf, pos, module_suffix);
    return (String)(*buf)[0:pos];
}

// Try to load one Flux LoRA site at (layer_prefix, layer, layer_suffix, module).
// Returns a present LoraTensor on success, or zero-init (present=false) on miss.
fn LoraTensor try_load_flux_site(
    DeviceContext* ctx,
    SafetensorsFile* sf,
    String layer_prefix,
    uint layer,
    String layer_suffix,
    String module_suffix,
    float global_alpha,
    float strength,
) {
    LoraTensor result;
    if (module_suffix.len == 0) return result;

    char[256] mod_buf;
    String full_module = flux_build_module_path(&mod_buf, layer_prefix, layer, layer_suffix, module_suffix);

    char[256] a_buf;
    char[256] b_buf;
    String a_name;
    String b_name;
    if (!find_lora_pair_names(sf, full_module, &a_buf, &b_buf, &a_name, &b_name)) {
        return result;
    }

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

    uint rank = (uint)a_tensor.shape[1];
    if (rank == 0) {
        a_tensor.free();
        b_tensor.free();
        return result;
    }

    float effective_alpha = try_load_per_tensor_alpha(sf, full_module);
    if (effective_alpha < 0.0f) effective_alpha = global_alpha;
    if (effective_alpha <= 0.0f) effective_alpha = (float)rank;

    result.a = a_tensor;
    result.b = b_tensor;
    result.rank = rank;
    result.scale = strength * effective_alpha / (float)rank;
    result.present = true;
    return result;
}

// Pick the first Flux scheme that yields any matches at layer 0 of either the
// dual or single stream. Returns null if no scheme matches.
fn FluxLoraScheme* detect_flux_lora_scheme(SafetensorsFile* sf) {
    for (uint i = 0; i < FLUX_LORA_SCHEMES.len; i++) {
        FluxLoraScheme* s = &FLUX_LORA_SCHEMES[i];
        String[*] dual_modules = {
            s.img_qkv, s.img_q, s.img_proj, s.img_mlp_up, s.txt_qkv, s.txt_q
        };
        String[*] single_modules = {
            s.single_linear1, s.single_q, s.single_linear2
        };
        for (uint m = 0; m < dual_modules.len; m++) {
            if (dual_modules[m].len == 0) continue;
            char[256] mod_buf;
            String full = flux_build_module_path(&mod_buf, s.dual_prefix, 0, s.layer_suffix, dual_modules[m]);
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
        for (uint m = 0; m < single_modules.len; m++) {
            if (single_modules[m].len == 0) continue;
            char[256] mod_buf;
            String full = flux_build_module_path(&mod_buf, s.single_prefix, 0, s.layer_suffix, single_modules[m]);
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

// Load a Flux LoRA safetensors file into a FluxLoraAdapter sized for the model's
// dual / single block counts. Layers absent from the LoRA file leave their
// LoraTensor.present flags false (a no-op at compute time).
fn FluxLoraAdapter? load_flux_lora_safetensors(
    DeviceContext* ctx,
    String path,
    float strength,
    uint n_dual,
    uint n_single,
) {
    io::printfn("  Loading Flux LoRA: %s", path);
    SafetensorsFile sf = safetensors_open(path)!;
    defer sf.close();

    float alpha = 0.0f;
    if (try v = sf.get_metadata("lora_alpha")) {
        alpha = (float)(v.to_double() ?? 0.0);
    } else if (try v = sf.get_metadata("ss_network_alpha")) {
        alpha = (float)(v.to_double() ?? 0.0);
    } else if (try v = sf.get_metadata("alpha")) {
        alpha = (float)(v.to_double() ?? 0.0);
    }
    if (alpha == 0.0f) {
        io::printfn("  Flux LoRA: no alpha in __metadata__, will default to rank per pair");
    }

    FluxLoraScheme* scheme = detect_flux_lora_scheme(&sf);
    if (scheme == null) {
        io::printfn("  Flux LoRA: no recognised tensor naming scheme. Header keys (first 8):");
        usz max = sf.n_tensors < 8 ? sf.n_tensors : 8;
        for (usz i = 0; i < max; i++) {
            io::printfn("    %s", sf.tensors[i].name);
        }
        return COMPUTE_ERROR~;
    }
    io::printfn("  Flux LoRA scheme: dual=\"%s\", single=\"%s\"", scheme.dual_prefix, scheme.single_prefix);

    FluxDualLoraLayer[]   dual   = n_dual   > 0 ? mem::new_array(FluxDualLoraLayer, n_dual)     : (FluxDualLoraLayer[]){};
    FluxSingleLoraLayer[] single = n_single > 0 ? mem::new_array(FluxSingleLoraLayer, n_single) : (FluxSingleLoraLayer[]){};

    uint covered = 0;

    for (uint l = 0; l < n_dual; l++) {
        FluxDualLoraLayer* dl = &dual[l];
        dl.img_qkv      = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_qkv,      alpha, strength);
        dl.img_q        = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_q,        alpha, strength);
        dl.img_k        = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_k,        alpha, strength);
        dl.img_v        = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_v,        alpha, strength);
        dl.img_proj     = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_proj,     alpha, strength);
        dl.img_mlp_up   = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_mlp_up,   alpha, strength);
        dl.img_mlp_down = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_mlp_down, alpha, strength);
        dl.txt_qkv      = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_qkv,      alpha, strength);
        dl.txt_q        = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_q,        alpha, strength);
        dl.txt_k        = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_k,        alpha, strength);
        dl.txt_v        = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_v,        alpha, strength);
        dl.txt_proj     = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_proj,     alpha, strength);
        dl.txt_mlp_up   = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_mlp_up,   alpha, strength);
        dl.txt_mlp_down = try_load_flux_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_mlp_down, alpha, strength);

        if (dl.img_qkv.present)      covered++;
        if (dl.img_q.present)        covered++;
        if (dl.img_k.present)        covered++;
        if (dl.img_v.present)        covered++;
        if (dl.img_proj.present)     covered++;
        if (dl.img_mlp_up.present)   covered++;
        if (dl.img_mlp_down.present) covered++;
        if (dl.txt_qkv.present)      covered++;
        if (dl.txt_q.present)        covered++;
        if (dl.txt_k.present)        covered++;
        if (dl.txt_v.present)        covered++;
        if (dl.txt_proj.present)     covered++;
        if (dl.txt_mlp_up.present)   covered++;
        if (dl.txt_mlp_down.present) covered++;
    }

    for (uint l = 0; l < n_single; l++) {
        FluxSingleLoraLayer* sl = &single[l];
        sl.linear1 = try_load_flux_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_linear1, alpha, strength);
        sl.q       = try_load_flux_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_q,       alpha, strength);
        sl.k       = try_load_flux_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_k,       alpha, strength);
        sl.v       = try_load_flux_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_v,       alpha, strength);
        sl.linear2 = try_load_flux_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_linear2, alpha, strength);

        if (sl.linear1.present) covered++;
        if (sl.q.present)       covered++;
        if (sl.k.present)       covered++;
        if (sl.v.present)       covered++;
        if (sl.linear2.present) covered++;
    }

    io::printfn("  Flux LoRA loaded: alpha=%.2f strength=%.2f, covered %d sites",
        alpha, strength, covered);

    if (covered == 0) {
        io::printfn("  Flux LoRA: 0 sites covered — first 10 file keys for diagnosis:");
        usz max = sf.n_tensors < 10 ? sf.n_tensors : 10;
        for (usz i = 0; i < max; i++) {
            io::printfn("    %s", sf.tensors[i].name);
        }
    }

    return {
        .path = path,
        .strength = strength,
        .alpha = alpha,
        .dual_layers = n_dual > 0 ? &dual[0] : null,
        .single_layers = n_single > 0 ? &single[0] : null,
        .n_dual = n_dual,
        .n_single = n_single,
        .covered_sites = covered,
        .loaded = true,
    };
}

// ============================================================================
// Flux 2 Klein LoKr (Low-rank Kronecker) adapter support.
//
// LyCORIS LoKr files (e.g. trained with ai-toolkit) decompose the delta as a
// Kronecker product instead of a low-rank product:
//   ΔW = scale * (W1 ⊗ W2)
// where W1 is small ([w1_rows, w1_cols], typically 4×4) and W2 is rectangular
// of shape [w2_rows, w2_cols], satisfying w1_rows * w2_rows = out_dim and
// w1_cols * w2_cols = in_dim.
//
// We only support the "full" variant where both lokr_w1 and lokr_w2 are stored
// as dense matrices (no further low-rank factorisation into _a / _b pairs).
// All sites observed in the Flux 2 Klein ai-toolkit LoKr files use this form.
//
// Naming uses scheme #2 (ComfyUI "diffusion_model." prefix) — no sub-range
// targeting (Q/K/V are always inside the fused qkv weight), so we only carry
// the fused module slots.
// ============================================================================

// One adapted linear in LoKr form. The two factor matrices are uploaded as
// F32 tensors; the kernel does the Kronecker expansion element-wise on the
// fly during the Q8_0 merge pass.
struct LokrTensor {
    Tensor w1;       // [w1_rows, w1_cols] F32 (after upload reversal: shape[0]=w1_cols, shape[1]=w1_rows)
    Tensor w2;       // [w2_rows, w2_cols] F32 (after upload reversal: shape[0]=w2_cols, shape[1]=w2_rows)
    uint   w1_rows;
    uint   w1_cols;
    uint   w2_rows;
    uint   w2_cols;
    float  scale;
    bool   present;
}

fn void LokrTensor.free(&self) {
    if (self.present) {
        self.w1.free();
        self.w2.free();
        self.present = false;
    }
}

struct FluxDualLokrLayer {
    LokrTensor img_qkv;
    LokrTensor img_proj;
    LokrTensor img_mlp_up;     // img_mlp.0
    LokrTensor img_mlp_down;   // img_mlp.2
    LokrTensor txt_qkv;
    LokrTensor txt_proj;
    LokrTensor txt_mlp_up;
    LokrTensor txt_mlp_down;
}

fn void FluxDualLokrLayer.free(&self) {
    self.img_qkv.free(); self.img_proj.free();
    self.img_mlp_up.free(); self.img_mlp_down.free();
    self.txt_qkv.free(); self.txt_proj.free();
    self.txt_mlp_up.free(); self.txt_mlp_down.free();
}

struct FluxSingleLokrLayer {
    LokrTensor linear1;
    LokrTensor linear2;
}

fn void FluxSingleLokrLayer.free(&self) {
    self.linear1.free(); self.linear2.free();
}

struct FluxLokrAdapter {
    String path;
    float  strength;
    FluxDualLokrLayer*   dual_layers;
    FluxSingleLokrLayer* single_layers;
    uint   n_dual;
    uint   n_single;
    uint   covered_sites;
    bool   loaded;
}

fn void FluxLokrAdapter.free(&self) {
    if (!self.loaded) return;
    for (uint l = 0; l < self.n_dual; l++)   self.dual_layers[l].free();
    for (uint l = 0; l < self.n_single; l++) self.single_layers[l].free();
    if (self.n_dual > 0)   mem::free(self.dual_layers);
    if (self.n_single > 0) mem::free(self.single_layers);
    self.loaded = false;
}

// Try to load one LoKr site (one lokr_w1 + one lokr_w2 + one alpha scalar)
// under module path "{prefix}{layer}{suffix}{module}". Returns a populated
// LokrTensor on success or zero-init (present=false) on miss.
fn LokrTensor try_load_flux_lokr_site(
    DeviceContext* ctx,
    SafetensorsFile* sf,
    String layer_prefix,
    uint layer,
    String layer_suffix,
    String module_suffix,
    float strength,
) {
    LokrTensor result;
    if (module_suffix.len == 0) return result;

    char[256] mod_buf;
    String full_module = flux_build_module_path(&mod_buf, layer_prefix, layer, layer_suffix, module_suffix);

    char[256] w1_buf;
    char[256] w2_buf;
    usz p1 = append_str(&w1_buf, 0, full_module);
    p1 = append_str(&w1_buf, p1, ".lokr_w1");
    usz p2 = append_str(&w2_buf, 0, full_module);
    p2 = append_str(&w2_buf, p2, ".lokr_w2");
    String w1_name = (String)w1_buf[0:p1];
    String w2_name = (String)w2_buf[0:p2];

    SafetensorsTensorInfo* w1_info;
    SafetensorsTensorInfo* w2_info;
    if (try info = sf.find_tensor(w1_name)) { w1_info = info; } else { return result; }
    if (try info = sf.find_tensor(w2_name)) { w2_info = info; } else { return result; }

    if (w1_info.n_dims != 2 || w2_info.n_dims != 2) return result;

    // Safetensors header shapes are in [rows, cols] = [out, in] order.
    uint w1_rows = (uint)w1_info.shape[0];
    uint w1_cols = (uint)w1_info.shape[1];
    uint w2_rows = (uint)w2_info.shape[0];
    uint w2_cols = (uint)w2_info.shape[1];

    Tensor w1_tensor;
    Tensor w2_tensor;
    if (try t = upload_safetensor_f32(ctx, sf, w1_name)) {
        w1_tensor = t;
    } else {
        return result;
    }
    if (try t = upload_safetensor_f32(ctx, sf, w2_name)) {
        w2_tensor = t;
    } else {
        w1_tensor.free();
        return result;
    }

    // Scale calibration: ai-toolkit LoKr files declare a per-site `.alpha`
    // tensor (often mislabelled as BF16 but actually F16-encoded) with values
    // around ~32. The LyCORIS canonical scale is `alpha / lora_dim`, but
    // lora_dim is not in metadata. Empirically, raw scale = strength (i.e.
    // alpha treated as already-normalised by the trainer) produces sensible
    // output for ai-toolkit's files.
    float scale = strength;

    result.w1 = w1_tensor;
    result.w2 = w2_tensor;
    result.w1_rows = w1_rows;
    result.w1_cols = w1_cols;
    result.w2_rows = w2_rows;
    result.w2_cols = w2_cols;
    result.scale = scale;
    result.present = true;
    return result;
}

// Probe the file for any `.lokr_w1` tensor under one of our known Flux schemes.
// Returns the matching scheme or null.
fn FluxLoraScheme* detect_flux_lokr_scheme(SafetensorsFile* sf) {
    String[*] dual_modules = { "img_attn.qkv", "img_attn.proj", "img_mlp.0", "txt_attn.qkv" };
    String[*] single_modules = { "linear1", "linear2" };
    for (uint i = 0; i < FLUX_LORA_SCHEMES.len; i++) {
        FluxLoraScheme* s = &FLUX_LORA_SCHEMES[i];
        // Only native-fused schemes have a meaningful LoKr layout — the
        // Diffusers scheme uses split Q/K/V and isn't observed in LoKr files.
        if (s.img_qkv.len == 0) continue;
        for (uint m = 0; m < dual_modules.len; m++) {
            char[256] mod_buf;
            String full = flux_build_module_path(&mod_buf, s.dual_prefix, 0, s.layer_suffix, dual_modules[m]);
            char[256] w_buf;
            usz pw = append_str(&w_buf, 0, full);
            pw = append_str(&w_buf, pw, ".lokr_w1");
            String w_name = (String)w_buf[0:pw];
            if (try _info = sf.find_tensor(w_name)) return s;
        }
        for (uint m = 0; m < single_modules.len; m++) {
            char[256] mod_buf;
            String full = flux_build_module_path(&mod_buf, s.single_prefix, 0, s.layer_suffix, single_modules[m]);
            char[256] w_buf;
            usz pw = append_str(&w_buf, 0, full);
            pw = append_str(&w_buf, pw, ".lokr_w1");
            String w_name = (String)w_buf[0:pw];
            if (try _info = sf.find_tensor(w_name)) return s;
        }
    }
    return null;
}

// Public entry point: open a safetensors file, autodetect the naming scheme,
// and populate a FluxLokrAdapter for the model's block counts.
fn FluxLokrAdapter? load_flux_lokr_safetensors(
    DeviceContext* ctx,
    String path,
    float strength,
    uint n_dual,
    uint n_single,
) {
    io::printfn("  Loading Flux LoKr: %s", path);
    SafetensorsFile sf = safetensors_open(path)!;
    defer sf.close();

    FluxLoraScheme* scheme = detect_flux_lokr_scheme(&sf);
    if (scheme == null) {
        io::printfn("  Flux LoKr: no recognised tensor naming scheme. Header keys (first 8):");
        usz max = sf.n_tensors < 8 ? sf.n_tensors : 8;
        for (usz i = 0; i < max; i++) {
            io::printfn("    %s", sf.tensors[i].name);
        }
        return COMPUTE_ERROR~;
    }
    io::printfn("  Flux LoKr scheme: dual=\"%s\", single=\"%s\"", scheme.dual_prefix, scheme.single_prefix);

    FluxDualLokrLayer[]   dual   = n_dual   > 0 ? mem::new_array(FluxDualLokrLayer, n_dual)     : (FluxDualLokrLayer[]){};
    FluxSingleLokrLayer[] single = n_single > 0 ? mem::new_array(FluxSingleLokrLayer, n_single) : (FluxSingleLokrLayer[]){};

    uint covered = 0;
    for (uint l = 0; l < n_dual; l++) {
        FluxDualLokrLayer* dl = &dual[l];
        dl.img_qkv      = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_qkv,      strength);
        dl.img_proj     = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_proj,     strength);
        dl.img_mlp_up   = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_mlp_up,   strength);
        dl.img_mlp_down = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.img_mlp_down, strength);
        dl.txt_qkv      = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_qkv,      strength);
        dl.txt_proj     = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_proj,     strength);
        dl.txt_mlp_up   = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_mlp_up,   strength);
        dl.txt_mlp_down = try_load_flux_lokr_site(ctx, &sf, scheme.dual_prefix, l, scheme.layer_suffix, scheme.txt_mlp_down, strength);

        if (dl.img_qkv.present)      covered++;
        if (dl.img_proj.present)     covered++;
        if (dl.img_mlp_up.present)   covered++;
        if (dl.img_mlp_down.present) covered++;
        if (dl.txt_qkv.present)      covered++;
        if (dl.txt_proj.present)     covered++;
        if (dl.txt_mlp_up.present)   covered++;
        if (dl.txt_mlp_down.present) covered++;
    }

    for (uint l = 0; l < n_single; l++) {
        FluxSingleLokrLayer* sl = &single[l];
        sl.linear1 = try_load_flux_lokr_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_linear1, strength);
        sl.linear2 = try_load_flux_lokr_site(ctx, &sf, scheme.single_prefix, l, scheme.layer_suffix, scheme.single_linear2, strength);

        if (sl.linear1.present) covered++;
        if (sl.linear2.present) covered++;
    }

    io::printfn("  Flux LoKr loaded: strength=%.2f, covered %d sites", strength, covered);

    if (covered == 0) {
        io::printfn("  Flux LoKr: 0 sites covered — first 10 file keys for diagnosis:");
        usz max = sf.n_tensors < 10 ? sf.n_tensors : 10;
        for (usz i = 0; i < max; i++) {
            io::printfn("    %s", sf.tensors[i].name);
        }
    }

    return {
        .path = path,
        .strength = strength,
        .dual_layers = n_dual > 0 ? &dual[0] : null,
        .single_layers = n_single > 0 ? &single[0] : null,
        .n_dual = n_dual,
        .n_single = n_single,
        .covered_sites = covered,
        .loaded = true,
    };
}
