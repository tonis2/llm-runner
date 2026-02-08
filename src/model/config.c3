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
    String output;
    String layer_prefix;
    String attn_norm;
    String wq;
    String wk;
    String wv;
    String wo;
    String ffn_norm;
    String ffn_gate;
    String ffn_up;
    String ffn_down;
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
    Object* root = json::parse_string(mem, json_text)!;

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

    uint head_dim = n_heads > 0 ? dim / n_heads : 0;

    ModelConfig config = {
        .name = name,
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

    io::printfn("\n=== Model Config ===");
    io::printfn("  architecture: %s", config.name);
    io::printfn("  norm: %s", norm_str);
    io::printfn("  ffn: %s", ffn_str);
    io::printfn("  dim: %d", config.dim);
    io::printfn("  n_layers: %d", config.n_layers);
    io::printfn("  n_heads: %d", config.n_heads);
    io::printfn("  n_kv_heads: %d", config.n_kv_heads);
    io::printfn("  ffn_dim: %d", config.ffn_dim);
    io::printfn("  vocab_size: %d", config.vocab_size);
    io::printfn("  head_dim: %d", config.head_dim);
    io::printfn("  rope_theta: %f", config.rope_theta);
    io::printfn("  rms_eps: %f", config.rms_eps);

    // Don't free root - strings in ModelConfig reference JSON-allocated memory
    return config;
}

fn WeightNames? load_weight_names(String json_text) {
    Object* root = json::parse_string(mem, json_text)!;

    Object* w = root.get("weights")!;

    WeightNames names = {
        .embedding = w.get_string("embedding") ?? "token_embd.weight",
        .output_norm = w.get_string("output_norm") ?? "output_norm.weight",
        .output = w.get_string("output") ?? "output.weight",
        .layer_prefix = w.get_string("layer_prefix") ?? "blk",
        .attn_norm = w.get_string("attn_norm") ?? "attn_norm.weight",
        .wq = w.get_string("attn_q") ?? "attn_q.weight",
        .wk = w.get_string("attn_k") ?? "attn_k.weight",
        .wv = w.get_string("attn_v") ?? "attn_v.weight",
        .wo = w.get_string("attn_output") ?? "attn_output.weight",
        .ffn_norm = w.get_string("ffn_norm") ?? "ffn_norm.weight",
        .ffn_gate = w.get_string("ffn_gate") ?? "ffn_gate.weight",
        .ffn_up = w.get_string("ffn_up") ?? "ffn_up.weight",
        .ffn_down = w.get_string("ffn_down") ?? "ffn_down.weight",
    };

    // Don't free root - strings in WeightNames reference JSON-allocated memory
    return names;
}
