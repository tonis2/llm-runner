module flux_example;

import std::io;
import std::io::file;
import std::encoding::json;
import std::collections::object;
import std::core::mem;
import flux;

faultdef ARGS_INVALID;

struct ServerOptions {
    bool   enabled;
    bool   keep_dit;
    String bind_addr;
    int    port;
}

fn ServerOptions server_opts_defaults() {
    return { .enabled = false, .keep_dit = false, .bind_addr = "127.0.0.1", .port = 7860 };
}

fn flux::FluxConfig? parse_flux_args(String[] args, ServerOptions* server_opts) {
    flux::FluxConfig config = flux::flux_config_defaults();
    *server_opts = server_opts_defaults();

    if (args.len < 2) {
        print_usage();
        return ARGS_INVALID~;
    }

    // Pre-scan for --config
    for (usz i = 1; i + 1 < args.len; i++) {
        if (args[i] == "--config") {
            load_flux_config(args[i + 1], &config);
            break;
        }
    }

    // Parse CLI args (override config file values)
    usz i = 1;
    while (i < args.len) {
        if (args[i].len > 2 && args[i][0] == '-' && args[i][1] == '-') {
            // Boolean flag (no value)
            if (args[i] == "--server") { server_opts.enabled = true; i++; continue; }
            if (args[i] == "--keep-dit") { server_opts.keep_dit = true; i++; continue; }

            if (i + 1 >= args.len) { i++; continue; }

            if (args[i] == "--model") { config.model_path = args[i + 1]; i += 2; }
            else if (args[i] == "--text-model") { config.text_model_path = args[i + 1]; i += 2; }
            else if (args[i] == "--vae") { config.vae_path = args[i + 1]; i += 2; }
            else if (args[i] == "--prompt") { config.prompt = args[i + 1]; i += 2; }
            else if (args[i] == "--output") { config.output_path = args[i + 1]; i += 2; }
            else if (args[i] == "--size") { config.image_size = args[i + 1].to_uint() ?? 1024; i += 2; }
            else if (args[i] == "--width") { config.image_width = args[i + 1].to_uint() ?? 0; i += 2; }
            else if (args[i] == "--height") { config.image_height = args[i + 1].to_uint() ?? 0; i += 2; }
            else if (args[i] == "--steps") { config.num_steps = args[i + 1].to_uint() ?? 4; i += 2; }
            else if (args[i] == "--seed") { config.seed = args[i + 1].to_uint() ?? 42; i += 2; }
            else if (args[i] == "--input") { config.input_path = args[i + 1]; i += 2; }
            else if (args[i] == "--edit-mode") { config.edit_mode = args[i + 1]; i += 2; }
            else if (args[i] == "--strength") { config.strength = args[i + 1].to_float() ?? 0.6f; i += 2; }
            else if (args[i] == "--lora") { config.lora_path = args[i + 1]; i += 2; }
            else if (args[i] == "--lora-strength") { config.lora_strength = args[i + 1].to_float() ?? 1.5f; i += 2; }
            else if (args[i] == "--bind") { server_opts.bind_addr = args[i + 1]; i += 2; }
            else if (args[i] == "--port") { server_opts.port = (int)(args[i + 1].to_uint() ?? 7860); i += 2; }
            else if (args[i] == "--config") { i += 2; }
            else { i++; }
        } else {
            i++;
        }
    }

    // Validate (prompt not required in server mode — comes per-request)
    if (config.model_path.len == 0) {
        io::printfn("Error: No DiT model path (--model or config 'model')");
        return ARGS_INVALID~;
    }
    if (config.text_model_path.len == 0) {
        io::printfn("Error: No text model path (--text-model or config 'text_model')");
        return ARGS_INVALID~;
    }
    if (config.vae_path.len == 0) {
        io::printfn("Error: No VAE path (--vae or config 'vae')");
        return ARGS_INVALID~;
    }
    if (!server_opts.enabled && config.prompt.len == 0) {
        io::printfn("Error: No prompt (--prompt or config 'prompt')");
        return ARGS_INVALID~;
    }

    return config;
}

fn void load_flux_config(String path, flux::FluxConfig* config) {
    char[] data = file::load(mem, path)!!;
    Object* cfg = json::parse(mem, (String)data)!!;

    if (try v = cfg.get_string("model")) config.model_path = v;
    if (try v = cfg.get_string("text_model")) config.text_model_path = v;
    if (try v = cfg.get_string("vae")) config.vae_path = v;
    if (try v = cfg.get_string("prompt")) config.prompt = v;
    if (try v = cfg.get_string("input")) config.input_path = v;
    if (try v = cfg.get_string("output")) config.output_path = v;
    if (try v = cfg.get_uint("steps")) config.num_steps = v;
    if (try v = cfg.get_float("strength")) config.strength = (float)v;
    if (try v = cfg.get_uint("seed")) config.seed = v;
    if (try v = cfg.get_uint("size")) config.image_size = v;
    if (try v = cfg.get_uint("width")) config.image_width = v;
    if (try v = cfg.get_uint("height")) config.image_height = v;
    if (try v = cfg.get_string("edit_mode")) config.edit_mode = v;

    // LoRA: accept either { "lora": "path.safetensors" } (strength defaults to 1.0)
    // or { "lora": { "path": "...", "strength": 0.8 } }
    if (try lv = cfg.get_string("lora")) {
        config.lora_path = lv;
    } else if (try lobj = cfg.get("lora")) {
        if (try p = lobj.get_string("path")) config.lora_path = p;
        if (try s = lobj.get_float("strength")) config.lora_strength = (float)s;
    }

    io::printfn("Flux config loaded from: %s", path);
}

fn void print_usage() {
    io::printfn("Usage: flux --config <config.json>");
    io::printfn("       flux --model <dit.gguf> --text-model <qwen3.gguf> --vae <vae.safetensors> --prompt \"...\"");
    io::printfn("");
    io::printfn("Options:");
    io::printfn("  --model        Flux DiT GGUF model");
    io::printfn("  --text-model   Qwen3-8B encoder GGUF");
    io::printfn("  --vae          Flux VAE safetensors");
    io::printfn("  --prompt       Text prompt");
    io::printfn("  --output       Output path (default: output.png)");
    io::printfn("  --size         Square image size (default: 1024)");
    io::printfn("  --width        Image width  (overrides --size for width)");
    io::printfn("  --height       Image height (overrides --size for height)");
    io::printfn("  --steps        Denoising steps (default: 4)");
    io::printfn("  --seed         Random seed (default: 42)");
    io::printfn("  --input        Input image (PNG or JPG) for img2img / kontext modes");
    io::printfn("  --edit-mode    Edit mode: img2img | kontext (default: txt2img)");
    io::printfn("  --strength     img2img strength 0..1 (default: 0.75)");
    io::printfn("  --lora         Optional LoRA safetensors file");
    io::printfn("  --lora-strength LoRA strength multiplier (default: 1.0)");
    io::printfn("  --config       Load from JSON config file");
    io::printfn("");
    io::printfn("Server mode:");
    io::printfn("  --server       Run as HTTP server on /sdapi/v1/img2img (POST)");
    io::printfn("  --bind         Bind address (default: 127.0.0.1)");
    io::printfn("  --port         Port (default: 7860)");
    io::printfn("  --keep-dit     Keep DiT weights resident in VRAM across requests (~9 GB; risky on 16 GB)");
}
