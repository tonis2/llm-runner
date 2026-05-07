module flux_example;

import std::io;
import std::io::file;
import std::encoding::json;
import std::collections::object;
import std::core::mem;
import flux;

faultdef ARGS_INVALID;

fn flux::FluxConfig? parse_flux_args(String[] args) {
    flux::FluxConfig config = flux::flux_config_defaults();

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
            if (args[i] == "--offload") { config.offload = true; i++; continue; }
            if (i + 1 >= args.len) { i++; continue; }

            if (args[i] == "--model") { config.model_path = args[i + 1]; i += 2; }
            else if (args[i] == "--text-model") { config.text_model_path = args[i + 1]; i += 2; }
            else if (args[i] == "--vae") { config.vae_path = args[i + 1]; i += 2; }
            else if (args[i] == "--prompt") { config.prompt = args[i + 1]; i += 2; }
            else if (args[i] == "--input") { config.input_path = args[i + 1]; i += 2; }
            else if (args[i] == "--output") { config.output_path = args[i + 1]; i += 2; }
            else if (args[i] == "--size") { config.image_size = args[i + 1].to_uint() ?? 1024; i += 2; }
            else if (args[i] == "--steps") { config.num_steps = args[i + 1].to_uint() ?? 4; i += 2; }
            else if (args[i] == "--cfg-scale") { config.cfg_scale = args[i + 1].to_float() ?? 1.0; i += 2; }
            else if (args[i] == "--strength") { config.strength = args[i + 1].to_float() ?? 0.75; i += 2; }
            else if (args[i] == "--seed") { config.seed = args[i + 1].to_uint() ?? 42; i += 2; }
            else if (args[i] == "--config") { i += 2; }
            else { i++; }
        } else {
            i++;
        }
    }

    // Validate
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
    if (config.prompt.len == 0) {
        io::printfn("Error: No prompt (--prompt or config 'prompt')");
        return ARGS_INVALID~;
    }

    return config;
}

fn void load_flux_config(String path, flux::FluxConfig* config) {
    char[] data = file::load(mem, path)!!;
    Object* cfg = json::parse_string(mem, (String)data)!!;

    if (try v = cfg.get_string("model")) config.model_path = v;
    if (try v = cfg.get_string("text_model")) config.text_model_path = v;
    if (try v = cfg.get_string("vae")) config.vae_path = v;
    if (try v = cfg.get_string("prompt")) config.prompt = v;
    if (try v = cfg.get_string("input")) config.input_path = v;
    if (try v = cfg.get_string("output")) config.output_path = v;
    if (try v = cfg.get_uint("steps")) config.num_steps = v;
    if (try v = cfg.get_float("cfg_scale")) config.cfg_scale = (float)v;
    if (try v = cfg.get_float("strength")) config.strength = (float)v;
    if (try v = cfg.get_uint("seed")) config.seed = v;
    if (try v = cfg.get_uint("size")) config.image_size = v;
    if (try v = cfg.get_bool("offload")) config.offload = v;

    io::printfn("Flux config loaded from: %s", path);
}

fn void print_usage() {
    io::printfn("Usage: flux --config <config.json>");
    io::printfn("       flux --model <dit.gguf> --text-model <qwen3.gguf> --vae <ae.safetensors> --prompt \"...\"");
    io::printfn("");
    io::printfn("Options:");
    io::printfn("  --model        Flux DiT GGUF model");
    io::printfn("  --text-model   Qwen3-8B encoder GGUF");
    io::printfn("  --vae          Flux VAE safetensors (ae.safetensors)");
    io::printfn("  --prompt       Text prompt");
    io::printfn("  --input        Input image for img2img editing");
    io::printfn("  --output       Output path (default: output.png)");
    io::printfn("  --size         Image size (default: 1024)");
    io::printfn("  --steps        Denoising steps (default: 4)");
    io::printfn("  --cfg-scale    Guidance scale (default: 1.0)");
    io::printfn("  --strength     Img2img strength 0-1 (default: 0.75)");
    io::printfn("  --seed         Random seed (default: 42)");
    io::printfn("  --offload      Stream DiT layers from CPU RAM");
    io::printfn("  --config       Load from JSON config file");
}
