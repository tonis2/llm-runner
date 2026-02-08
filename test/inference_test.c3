module llm;

import vk;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::core::test;
import std::math::random;

fn void test_stories260k_forward() @test {
    String model_path = "test/models/stories260K.gguf";

    // Memory map the GGUF file
    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();

    GGUFFile gf = gguf_parse(data)!!;
    defer gf.free();

    // Detect architecture and load config
    String arch_name = detect_architecture(&gf);
    test::eq("llama", arch_name);

    String config_json = (String)&LLAMA_CONFIG_JSON;
    ModelConfig config = load_arch_config(config_json, &gf)!!;
    WeightNames names = load_weight_names(config_json)!!;

    // Verify model dimensions
    test::@check(config.dim > 0, "dim should be > 0");
    test::@check(config.n_layers > 0, "n_layers should be > 0");
    test::@check(config.vocab_size > 0, "vocab_size should be > 0");

    // Create Vulkan context and load model
    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    Model model = load_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    Tokenizer tok = load_tokenizer(&gf)!!;
    defer tok.free();

    // Forward pass: BOS token at position 0
    model.forward(tok.bos_id, 0)!!;

    // Greedy sample the first token
    SamplingParams greedy = { .temperature = 0.0f, .top_k = 0, .top_p = 1.0f };
    random::Sfc64Random rng;
    random::seed(&rng, 42);
    uint first_token = sample_token(&ctx, &model.acts.logits, config.vocab_size, &greedy, &rng)!!;

    // The greedy first token should be deterministic
    test::@check(first_token > 0 && first_token < config.vocab_size,
        "first token %d should be valid (0 < id < %d)", first_token, config.vocab_size);

    io::printfn("  stories260K: BOS -> token %d (\"%s\")",
        first_token, tok.vocab[first_token]);

    // Run a second forward pass to verify no crash
    model.forward(first_token, 1)!!;
    uint second_token = sample_token(&ctx, &model.acts.logits, config.vocab_size, &greedy, &rng)!!;
    test::@check(second_token < config.vocab_size,
        "second token %d should be valid", second_token);

    io::printfn("  stories260K: token %d -> token %d (\"%s\")",
        first_token, second_token, tok.vocab[second_token]);
}
