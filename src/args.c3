module llm::args;

import std::io;
import std::io::file;
import std::encoding::json;
import std::collections::object;
import std::core::mem;

faultdef ARGS_INVALID,
         ARGS_MISSING_MODEL,
         ARGS_MISSING_PROMPT;

// Configuration for image generation
struct AppConfig {
    String model_path;
    String prompt;
    String output_path;
    String input_path;       // For img2img
    String vae_path;
    String taesd_path;
    String text_model_path;
    uint image_size;
    uint num_steps;
    float cfg_scale;
    uint seed;
}

// Parse arguments and config files
fn AppConfig? parse_args(String[] args) {
    AppConfig config = {
        .model_path = "",
        .prompt = "",
        .output_path = "output.png",
        .input_path = "",
        .vae_path = "",
        .taesd_path = "",
        .text_model_path = "",
        .image_size = 512,
        .num_steps = 4,
        .cfg_scale = 7.0,
        .seed = 42,
    };

    if (args.len < 2) {
        print_usage();
        return ARGS_INVALID~;
    }

    // Pre-scan for --config
    for (usz i = 1; i + 1 < args.len; i++) {
        if (args[i] == "--config") {
            if (!load_config(args[i + 1], &config)) {
                return ARGS_INVALID~;
            }
            break;
        }
    }

    // Parse CLI args
    usz i = 1;
    while (i < args.len) {
        if (args[i].len > 2 && args[i][0] == '-' && args[i][1] == '-') {
            if (i + 1 >= args.len) { i++; continue; }

            if (args[i] == "--output") {
                config.output_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--input") {
                config.input_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--vae") {
                config.vae_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--taesd") {
                config.taesd_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--text-model") {
                config.text_model_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--size") {
                config.image_size = args[i + 1].to_uint() ?? 512;
                i += 2;
            } else if (args[i] == "--steps") {
                config.num_steps = args[i + 1].to_uint() ?? 4;
                i += 2;
            } else if (args[i] == "--cfg-scale") {
                config.cfg_scale = args[i + 1].to_float() ?? 7.0;
                i += 2;
            } else if (args[i] == "--seed") {
                config.seed = args[i + 1].to_uint() ?? 42;
                i += 2;
            } else if (args[i] == "--config") {
                i += 2;  // Already handled
            } else {
                i++;
            }
        } else if (config.model_path.len == 0) {
            config.model_path = args[i];
            i++;
        } else if (config.prompt.len == 0) {
            config.prompt = args[i];
            i++;
        } else {
            i++;
        }
    }

    // Validate
    if (config.model_path.len == 0) {
        io::printfn("Error: No model path provided");
        print_usage();
        return ARGS_INVALID~;
    }

    if (config.prompt.len == 0) {
        io::printfn("Error: No prompt provided");
        print_usage();
        return ARGS_INVALID~;
    }

    return config;
}

fn bool load_config(String config_path, AppConfig* config) {
    char[] config_data = file::load(mem, config_path)!!;
    Object* cfg = json::parse_string(mem, (String)config_data)!!;

    if (try v = cfg.get_string("model")) config.model_path = v;
    if (try v = cfg.get_string("prompt")) config.prompt = v;
    if (try v = cfg.get_string("vae")) config.vae_path = v;
    if (try v = cfg.get_string("taesd")) config.taesd_path = v;
    if (try v = cfg.get_string("text_model")) config.text_model_path = v;
    if (try v = cfg.get_string("output")) config.output_path = v;
    if (try v = cfg.get_string("input")) config.input_path = v;
    if (try v = cfg.get_uint("steps")) config.num_steps = v;
    if (try v = cfg.get_float("cfg_scale")) config.cfg_scale = (float)v;
    if (try v = cfg.get_uint("seed")) config.seed = v;
    if (try v = cfg.get_uint("size")) config.image_size = v;

    io::printfn("Config loaded from: %s", config_path);
    return true;
}

fn void print_usage() {
    io::printfn("Usage: llm <model.gguf> <prompt> [options]");
    io::printfn("       llm --config <config.json>");
    io::printfn("");
    io::printfn("Options:");
    io::printfn("  --output       Output image path (default: output.png)");
    io::printfn("  --input        Input image for img2img mode");
    io::printfn("  --vae          Path to VAE safetensors");
    io::printfn("  --taesd        Path to TAESD safetensors");
    io::printfn("  --text-model   Path to text encoder GGUF");
    io::printfn("  --size         Image size (default: 512)");
    io::printfn("  --steps        Number of steps (default: 4)");
    io::printfn("  --cfg-scale    CFG scale (default: 7.0)");
    io::printfn("  --seed         Random seed (default: 42)");
    io::printfn("  --config       Load from JSON config file");
}
