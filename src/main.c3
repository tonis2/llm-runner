module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;
import std::time::clock;
import std::encoding::json;
import std::collections::object;

enum RunMode : char { LLM, TXT2IMG, IMG2IMG }

fn bool is_diffusion_model(GGUFFile* gf) {
    // Check for diffusion tensor name prefixes
    foreach (&t : gf.tensors) {
        if (t.name.starts_with("model.diffusion_model.") ||
            t.name.starts_with("cond_stage_model.") ||
            t.name.starts_with("first_stage_model.")) {
            return true;
        }
    }
    return false;
}

fn int main(String[] args) {
    if (args.len < 2) {
        io::printfn("Usage: llm <model.gguf> [prompt] [options]");
        io::printfn("       llm --config <config.json> [prompt]");
        io::printfn("");
        io::printfn("General options:");
        io::printfn("  --config   Load settings from JSON config file");
        io::printfn("");
        io::printfn("LLM options:");
        io::printfn("  --temp   Temperature (0 = greedy, default 0.7)");
        io::printfn("  --top-k  Top-k filtering (default 40)");
        io::printfn("  --top-p  Nucleus sampling threshold (default 0.9)");
        io::printfn("  --arch   Path to architecture config JSON");
        io::printfn("");
        io::printfn("Diffusion options:");
        io::printfn("  --mode     txt2img|img2img (auto-detected from model)");
        io::printfn("  --steps    Number of diffusion steps (default 1)");
        io::printfn("  --cfg-scale  Classifier-free guidance scale (default 1.0)");
        io::printfn("  --seed     Random seed (default 42)");
        io::printfn("  --output   Output image path (default output.ppm)");
        io::printfn("  --input    Input image for img2img mode");
        io::printfn("  --vae        Path to VAE safetensors (for DiT models)");
        io::printfn("  --taesd      Path to TAESD safetensors (lightweight VAE alternative)");
        io::printfn("  --text-model Path to text encoder GGUF (for DiT models)");
        io::printfn("  --size       Image size (default 512, for DiT models)");
        return 1;
    }

    // Parse arguments: positional (model, prompt) + flags
    String model_path = "";
    String prompt = "";
    String arch_path = "";
    SamplingParams sampling = { .temperature = 0.7f, .top_k = 40, .top_p = 0.9f };

    // Diffusion params
    RunMode run_mode = LLM;
    bool mode_forced = false;
    uint diff_steps = 1;
    float cfg_scale = 1.0f;
    ulong seed = 42;
    String output_path = "output.ppm";
    String input_path = "";
    String vae_path = "";
    String taesd_path = "";
    String text_model_path = "";
    uint image_size = 512;

    // Pre-scan for --config and load JSON defaults
    for (usz ci = 1; ci + 1 < args.len; ci++) {
        if (args[ci] == "--config") {
            String config_path = args[ci + 1];
            char[] config_data = file::load(mem, config_path)!!;
            Object* cfg = json::parse_string(mem, (String)config_data)!!;
            // Note: don't free cfg — string values reference its memory

            if (try v = cfg.get_string("model")) model_path = v;
            if (try v = cfg.get_string("prompt")) prompt = v;
            if (try v = cfg.get_string("arch")) arch_path = v;
            if (try v = cfg.get_string("vae")) vae_path = v;
            if (try v = cfg.get_string("taesd")) taesd_path = v;
            if (try v = cfg.get_string("text_model")) text_model_path = v;
            if (try v = cfg.get_string("output")) output_path = v;
            if (try v = cfg.get_string("input")) input_path = v;
            if (try v = cfg.get_float("temp")) sampling.temperature = (float)v;
            if (try v = cfg.get_uint("top_k")) sampling.top_k = v;
            if (try v = cfg.get_float("top_p")) sampling.top_p = (float)v;
            if (try v = cfg.get_uint("steps")) diff_steps = v;
            if (try v = cfg.get_float("cfg_scale")) cfg_scale = (float)v;
            if (try v = cfg.get_uint("seed")) seed = (ulong)v;
            if (try v = cfg.get_uint("size")) image_size = v;
            if (try v = cfg.get_string("mode")) {
                if (v == "txt2img") { run_mode = TXT2IMG; mode_forced = true; }
                else if (v == "img2img") { run_mode = IMG2IMG; mode_forced = true; }
            }

            io::printfn("Config loaded from: %s", config_path);
            break;
        }
    }

    // CLI args override config values
    usz i = 1;
    while (i < args.len) {
        if (args[i].len > 2 && args[i][0] == '-' && args[i][1] == '-') {
            if (i + 1 >= args.len) { i++; continue; }
            if (args[i] == "--temp") {
                sampling.temperature = args[i + 1].to_float() ?? 0.7f;
                i += 2;
            } else if (args[i] == "--top-k") {
                sampling.top_k = args[i + 1].to_uint() ?? 40;
                i += 2;
            } else if (args[i] == "--top-p") {
                sampling.top_p = args[i + 1].to_float() ?? 0.9f;
                i += 2;
            } else if (args[i] == "--arch") {
                arch_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--mode") {
                if (args[i + 1] == "txt2img") {
                    run_mode = TXT2IMG;
                    mode_forced = true;
                } else if (args[i + 1] == "img2img") {
                    run_mode = IMG2IMG;
                    mode_forced = true;
                }
                i += 2;
            } else if (args[i] == "--steps") {
                diff_steps = args[i + 1].to_uint() ?? 1;
                i += 2;
            } else if (args[i] == "--cfg-scale") {
                cfg_scale = args[i + 1].to_float() ?? 1.0f;
                i += 2;
            } else if (args[i] == "--seed") {
                seed = args[i + 1].to_ulong() ?? 42;
                i += 2;
            } else if (args[i] == "--output") {
                output_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--input") {
                input_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--vae") {
                vae_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--taesd") {
                taesd_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--text-model") {
                text_model_path = args[i + 1];
                i += 2;
            } else if (args[i] == "--size") {
                image_size = args[i + 1].to_uint() ?? 512;
                i += 2;
            } else if (args[i] == "--config") {
                i += 2;  // already handled in pre-scan
            } else {
                i++;
            }
        } else if (model_path.len == 0) {
            model_path = args[i];
            i++;
        } else if (prompt.len == 0) {
            prompt = args[i];
            i++;
        } else {
            i++;
        }
    }

    if (model_path.len == 0) {
        io::printfn("Error: no model path provided");
        return 1;
    }

    io::printfn("Loading model from: %s", model_path);

    // Memory map the GGUF file
    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();
    io::printfn("File mapped: %d bytes", data.len);

    GGUFFile gf = gguf_parse(data)!!;
    defer gf.free();
    gguf_print_info(&gf);

    // Detect model type: DiT, diffusion, or LLM
    bool is_dit = is_dit_model(&gf);
    bool is_diffusion = is_diffusion_model(&gf);

    if ((is_diffusion || is_dit) && !mode_forced) {
        run_mode = TXT2IMG;
    }

    // DiT / Z-Image pipeline
    if (is_dit && (run_mode == TXT2IMG || run_mode == IMG2IMG)) {
        io::printfn("DiT model detected — Z-Image pipeline");
        if (prompt.len == 0) {
            io::printfn("Error: a prompt is required for image generation");
            return 1;
        }

        DeviceContext ctx = createContext()!!;
        ctx.allocator.queue = &ctx.compute_queue;
        defer ctx.free();

        ZImageParams zparams = {
            .num_steps = diff_steps > 1 ? diff_steps : 8,
            .seed = (uint)seed,
            .output_path = output_path,
            .vae_path = vae_path,
            .taesd_path = taesd_path,
            .text_model_path = text_model_path,
            .image_size = image_size,
        };

        bool ok = run_zimage_pipeline(&ctx, &gf, prompt, &zparams);
        return ok ? 0 : 1;
    }

    if (run_mode == TXT2IMG || run_mode == IMG2IMG) {
        io::printfn("Diffusion model detected");
        io::printfn("Mode: %s, Steps: %d, CFG: %.1f, Seed: %d",
            run_mode == TXT2IMG ? "txt2img" : "img2img", diff_steps, cfg_scale, seed);
        io::printfn("Output: %s", output_path);

        if (prompt.len == 0) {
            io::printfn("Error: a prompt is required for image generation");
            return 1;
        }

        DeviceContext ctx = createContext()!!;
        ctx.allocator.queue = &ctx.compute_queue;  // Fix dangling pointer from createContext
        defer ctx.free();

        DiffusionPipeline pipeline = load_diffusion_pipeline(&ctx, &gf)!!;
        pipeline.fixup_pointers();  // Fix dangling kernel pointers from return-by-value
        defer pipeline.free();

        // Load CLIP tokenizer - SDXS doesn't have tokenizer metadata in GGUF
        Tokenizer tok;
        bool has_tokenizer = false;
        if (try t = load_tokenizer(&gf)) {
            tok = t;
            has_tokenizer = true;
        } else {
            io::printfn("Warning: No tokenizer in model. Using dummy tokens (prompt will be ignored).");
        }
        defer if (has_tokenizer) tok.free();

        DiffusionParams params = {
            .num_steps = diff_steps,
            .cfg_scale = cfg_scale,
            .seed = (uint)seed,
            .strength = 0.75f,
            .output_path = output_path,
        };

        bool ok;
        if (run_mode == TXT2IMG) {
            ok = pipeline.txt2img(has_tokenizer ? &tok : null, prompt, &params);
        } else {
            if (input_path.len == 0) {
                io::printfn("Error: --input required for img2img mode");
                return 1;
            }
            ok = pipeline.img2img(has_tokenizer ? &tok : null, prompt, input_path, &params);
        }

        return ok ? 0 : 1;
    }

    // LLM path
    String arch_name = detect_architecture(&gf);
    io::printfn("Architecture: %s", arch_name);

    String config_json;
    if (arch_path.len > 0) {
        // Load custom config from file
        io::printfn("Loading config from: %s", arch_path);
        config_json = (String)file::load(mem, arch_path)!!;
    } else if (arch_name == "llama") {
        config_json = (String)&LLAMA_CONFIG_JSON;
    } else if (arch_name == "qwen2") {
        config_json = (String)&QWEN2_CONFIG_JSON;
    } else if (arch_name == "qwen3") {
        config_json = (String)&QWEN3_CONFIG_JSON;
    } else if (arch_name == "phi3") {
        config_json = (String)&PHI3_CONFIG_JSON;
    } else {
        io::printfn("Error: unsupported architecture '%s'. Use --arch to provide a config.", arch_name);
        return 1;
    }

    ModelConfig config = load_arch_config(config_json, &gf)!!;
    WeightNames names = load_weight_names(config_json)!!;

    DeviceContext ctx = createContext()!!;
    ctx.allocator.queue = &ctx.compute_queue;  // Fix dangling pointer from createContext
    defer ctx.free();

    LlmModel model = load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    Tokenizer tok = load_tokenizer(&gf)!!;
    defer tok.free();

    // Encode prompt if provided
    uint[] prompt_tokens;
    usz n_prompt = 0;
    if (prompt.len > 0) {
        prompt_tokens = tok.encode(prompt)!!;
        n_prompt = prompt_tokens.len;
        io::printfn("Prompt encoded to %d tokens", n_prompt);
    }

    uint max_tokens = MAX_SEQ_LEN - (uint)n_prompt - 1;

    if (sampling.temperature <= 0.0f) {
        io::printfn("\nGenerating (greedy, max %d tokens)...\n", max_tokens);
    } else {
        io::printfn("\nGenerating (temp=%.1f, top_k=%d, top_p=%.1f, max %d tokens)...\n",
            sampling.temperature, sampling.top_k, sampling.top_p, max_tokens);
    }

    // Echo prompt text
    if (prompt.len > 0) {
        io::printf("%s", prompt);
    }

    // Prefill: BOS + prompt tokens
    Clock prefill_start = clock::now();
    model.forward(tok.bos_id, 0)!!;  // BOS at position 0
    for (usz pi = 0; pi < n_prompt; pi++) {
        model.forward(prompt_tokens[pi], (uint)(pi + 1))!!;
    }
    double prefill_sec = prefill_start.to_now().to_sec();

    // Decode: sample and generate
    random::Sfc64Random rng;
    random::seed(&rng, (ulong)42);
    uint next_pos = (uint)(n_prompt + 1);
    uint n_decoded = 0;
    char[256] decode_buf;
    Clock decode_start = clock::now();
    for (uint g = 0; g < max_tokens; g++) {
        uint token = sample_token(&ctx, &model.acts.logits, model.config.vocab_size, &sampling, &rng)!!;
        if (token == tok.eos_id) break;
        String text = tok.decode_token(token, &decode_buf);
        if (text.len > 0) {
            io::printf("%s", text);
        }
        model.forward(token, next_pos)!!;
        next_pos++;
        n_decoded++;
    }
    double decode_sec = decode_start.to_now().to_sec();
    io::printfn("");

    // Performance stats
    uint n_prefill = (uint)n_prompt + 1;  // BOS + prompt tokens
    io::printfn("\n--- Performance ---");
    io::printfn("Prefill: %d tokens in %.2f s (%.1f tok/s)", n_prefill, prefill_sec,
        (double)n_prefill / prefill_sec);
    io::printfn("Decode:  %d tokens in %.2f s (%.1f tok/s)", n_decoded, decode_sec,
        (double)n_decoded / decode_sec);
    io::printfn("Total:   %.2f s", prefill_sec + decode_sec);

    if (n_prompt > 0) mem::free(prompt_tokens);

    return 0;
}
