module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;

fn int main(String[] args) {
    if (args.len < 2) {
        io::printfn("Usage: llm <model.gguf> [prompt] [--temp T] [--top-k K] [--top-p P] [--arch config.json]");
        io::printfn("  --temp   Temperature (0 = greedy, default 0.7)");
        io::printfn("  --top-k  Top-k filtering (default 40)");
        io::printfn("  --top-p  Nucleus sampling threshold (default 0.9)");
        io::printfn("  --arch   Path to architecture config JSON (auto-detected if omitted)");
        return 1;
    }

    // Parse arguments: positional (model, prompt) + flags
    String model_path = "";
    String prompt = "";
    String arch_path = "";
    SamplingParams sampling = { .temperature = 0.7f, .top_k = 40, .top_p = 0.9f };

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

    // Detect architecture and load config
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
    } else if (arch_name == "phi3") {
        config_json = (String)&PHI3_CONFIG_JSON;
    } else {
        io::printfn("Error: unsupported architecture '%s'. Use --arch to provide a config.", arch_name);
        return 1;
    }

    ModelConfig config = load_arch_config(config_json, &gf)!!;
    WeightNames names = load_weight_names(config_json)!!;

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    Model model = load_model(&ctx, &gf, &config, &names)!!;
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
    model.forward(tok.bos_id, 0)!!;  // BOS at position 0
    for (usz pi = 0; pi < n_prompt; pi++) {
        model.forward(prompt_tokens[pi], (uint)(pi + 1))!!;
    }

    // Decode: sample and generate
    random::Sfc64Random rng;
    random::seed(&rng, (ulong)42);
    uint next_pos = (uint)(n_prompt + 1);
    char[256] decode_buf;
    for (uint g = 0; g < max_tokens; g++) {
        uint token = sample_token(&ctx, &model.acts.logits, model.config.vocab_size, &sampling, &rng)!!;
        if (token == tok.eos_id) break;
        String text = tok.decode_token(token, &decode_buf);
        if (text.len > 0) {
            io::printf("%s", text);
        }
        model.forward(token, next_pos)!!;
        next_pos++;
    }
    io::printfn("");

    if (n_prompt > 0) mem::free(prompt_tokens);

    return 0;
}
