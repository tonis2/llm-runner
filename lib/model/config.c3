module llm;

import std::io;
import std::core::mem;
import std::encoding::json;
import std::collections::object;

enum NormType : char { RMSNORM, LAYERNORM }
enum FfnType : char { SWIGLU, GELU_MLP }

struct ModelConfig {
    String name;
    NormType norm_type;
    FfnType ffn_type;
    uint dim;
    uint n_heads;
    uint n_kv_heads;
    uint n_layers;
    uint ffn_dim;
    uint vocab_size;
    uint head_dim;
    float rope_theta;
    float rms_eps;
}

struct WeightNames {
    String embedding;
    String output_norm;
    String output_norm_bias;
    String output;
    String layer_prefix;
    String attn_norm;
    String attn_norm_bias;
    String wq;
    String wk;
    String wv;
    String wo;
    String wq_bias;
    String wk_bias;
    String wv_bias;
    String wq_norm;
    String wk_norm;
    String ffn_norm;
    String ffn_norm_bias;
    String ffn_gate;
    String ffn_up;
    String ffn_down;
}

// The parsed JSON tree is released before these loaders return, so any string
// kept in ModelConfig or WeightNames has to be copied out of it first.
//
// An absent entry stays the null slice rather than becoming a copy of "": the
// fallbacks below are string literals living in rodata, and handing one of
// those to mem::free would corrupt the heap. Callers already test these with
// `.len > 0`, which reads the same either way.
fn String own_string(String s) @local => s.len > 0 ? s.copy(mem) : (String){};

fn void disown_string(String* s) @local {
    if ((*s).len > 0) mem::free(*s);
    *s = (String){};
}

fn String detect_architecture(GGUFFile* gf) {
    return gf.get_string("general.architecture") ?? "llama";
}

fn uint read_param_u32(Object* params, String param_name, GGUFFile* gf) {
    if (try param_obj = params.get(param_name)) {
        String key = param_obj.get_string("key") ?? "";
        uint def = (uint)(param_obj.get_float("default") ?? 0.0);
        if (key.len > 0) {
            return gf.get_u32(key) ?? def;
        }
        return def;
    }
    return 0;
}

fn float read_param_f32(Object* params, String param_name, GGUFFile* gf) {
    if (try param_obj = params.get(param_name)) {
        String key = param_obj.get_string("key") ?? "";
        float def = (float)(param_obj.get_float("default") ?? 0.0);
        if (key.len > 0) {
            return gf.get_f32(key) ?? def;
        }
        return def;
    }
    return 0.0;
}

fn ModelConfig? load_arch_config(String json_text, GGUFFile* gf) {
    Object* root = json::parse(mem, json_text)!;
    defer root.free();

    String name = root.get_string("name") ?? "unknown";

    // Parse norm type
    String norm_str = root.get_string("norm") ?? "rmsnorm";
    NormType norm_type = norm_str == "layernorm" ? LAYERNORM : RMSNORM;

    // Parse FFN type
    String ffn_str = root.get_string("ffn") ?? "swiglu";
    FfnType ffn_type = ffn_str == "gelu_mlp" ? GELU_MLP : SWIGLU;

    // Read params from GGUF using JSON-defined keys
    Object* params = root.get("params")!;

    uint dim = read_param_u32(params, "dim", gf);
    uint n_heads = read_param_u32(params, "n_heads", gf);
    uint n_kv_heads = read_param_u32(params, "n_kv_heads", gf);
    uint n_layers = read_param_u32(params, "n_layers", gf);
    uint ffn_dim = read_param_u32(params, "ffn_dim", gf);
    float rope_theta = read_param_f32(params, "rope_theta", gf);
    float rms_eps = read_param_f32(params, "rms_eps", gf);

    // Vocab size from embedding tensor shape
    uint vocab_size = 0;
    if (try weights_obj = root.get("weights")) {
        String emb_name = weights_obj.get_string("embedding") ?? "token_embd.weight";
        if (try emb = gf.find_tensor(emb_name)) {
            vocab_size = (uint)emb.shape[1];
        }
    }

    // head_dim: try GGUF metadata first (e.g. qwen3.attention.key_length=128),
    // fall back to dim/n_heads. Qwen3-4B has dim=2560, n_heads=32, head_dim=128 (not 80).
    uint head_dim = read_param_u32(params, "head_dim", gf);
    if (head_dim == 0) {
        head_dim = n_heads > 0 ? dim / n_heads : 0;
    }

    ModelConfig config = {
        .name = own_string(name),
        .norm_type = norm_type,
        .ffn_type = ffn_type,
        .dim = dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .n_layers = n_layers,
        .ffn_dim = ffn_dim,
        .vocab_size = vocab_size,
        .head_dim = head_dim,
        .rope_theta = rope_theta,
        .rms_eps = rms_eps,
    };


    return config;
}

fn void ModelConfig.free(&self) {
    disown_string(&self.name);
}

fn WeightNames? load_weight_names(String json_text) {
    Object* root = json::parse(mem, json_text)!;
    defer root.free();

    Object* w = root.get("weights")!;

    WeightNames names = {
        .embedding = own_string(w.get_string("embedding") ?? "token_embd.weight"),
        .output_norm = own_string(w.get_string("output_norm") ?? "output_norm.weight"),
        .output_norm_bias = own_string(w.get_string("output_norm_bias") ?? ""),
        .output = own_string(w.get_string("output") ?? "output.weight"),
        .layer_prefix = own_string(w.get_string("layer_prefix") ?? "blk"),
        .attn_norm = own_string(w.get_string("attn_norm") ?? "attn_norm.weight"),
        .attn_norm_bias = own_string(w.get_string("attn_norm_bias") ?? ""),
        .wq = own_string(w.get_string("attn_q") ?? "attn_q.weight"),
        .wk = own_string(w.get_string("attn_k") ?? "attn_k.weight"),
        .wv = own_string(w.get_string("attn_v") ?? "attn_v.weight"),
        .wo = own_string(w.get_string("attn_output") ?? "attn_output.weight"),
        .wq_bias = own_string(w.get_string("attn_q_bias") ?? ""),
        .wk_bias = own_string(w.get_string("attn_k_bias") ?? ""),
        .wv_bias = own_string(w.get_string("attn_v_bias") ?? ""),
        .wq_norm = own_string(w.get_string("attn_q_norm") ?? ""),
        .wk_norm = own_string(w.get_string("attn_k_norm") ?? ""),
        .ffn_norm = own_string(w.get_string("ffn_norm") ?? "ffn_norm.weight"),
        .ffn_norm_bias = own_string(w.get_string("ffn_norm_bias") ?? ""),
        .ffn_gate = own_string(w.get_string("ffn_gate") ?? "ffn_gate.weight"),
        .ffn_up = own_string(w.get_string("ffn_up") ?? "ffn_up.weight"),
        .ffn_down = own_string(w.get_string("ffn_down") ?? "ffn_down.weight"),
    };

    return names;
}

<*
 Release the names copied out of the config JSON.

 LlmModel keeps a *pointer* to a WeightNames in encode_only mode, for streaming
 layer loads — so this must outlive any model built from it. Declaring it before
 the model and deferring both gives that ordering for free.
*>
fn void WeightNames.free(&self) {
    String*[21] fields = {
        &self.embedding, &self.output_norm, &self.output_norm_bias, &self.output,
        &self.layer_prefix, &self.attn_norm, &self.attn_norm_bias,
        &self.wq, &self.wk, &self.wv, &self.wo,
        &self.wq_bias, &self.wk_bias, &self.wv_bias,
        &self.wq_norm, &self.wk_norm,
        &self.ffn_norm, &self.ffn_norm_bias,
        &self.ffn_gate, &self.ffn_up, &self.ffn_down,
    };
    foreach (field : fields) disown_string(field);
}
