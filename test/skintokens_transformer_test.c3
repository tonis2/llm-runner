module llm;

import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::core::test;

// The transformer half of SkinTokens is stock Qwen3 at hidden 896 with a
// vocabulary of skin tokens instead of text, so the converter renames it to
// llama.cpp's scheme and the engine's existing loader takes it from there.
// This is the test that that claim is true rather than plausible.
const String SKINTOKENS_GGUF = "test/models/skintokens/grpo_1400.gguf";

fn bool have_gguf()
{
    if (try File f = file::open(SKINTOKENS_GGUF, "rb"))
    {
        (void)f.close();
        return true;
    }
    io::printfn("  skipped: %s not present", SKINTOKENS_GGUF);
    return false;
}

fn void test_skintokens_transformer_forward() @test
{
    if (!have_gguf()) return;

    mmap::FileMmap weights = file::mmap_open(SKINTOKENS_GGUF, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    // Nothing here is SkinTokens-aware: the config is the engine's own qwen3
    // description, read against metadata the converter wrote.
    String schema = (String)&llm_text::QWEN3_CONFIG_JSON;
    ModelConfig config = load_arch_config(schema, &gf)!!;
    defer config.free();
    WeightNames names = load_weight_names(schema)!!;
    defer names.free();

    test::eq(896, (int)config.dim);
    test::eq(28, (int)config.n_layers);
    test::eq(33036, (int)config.vocab_size);

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    llm_text::LlmModel model = llm_text::load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    // The vocabulary is 32768 skin tokens, 267 tokenizer tokens and one global
    // EOS. Token 1 is a coordinate bin, which is a legal thing to feed even
    // without the mesh prefix the real pipeline puts in front.
    model.forward(1, 0)!!;
    float[] first = download_tensor(mem, &ctx, &model.acts.logits, config.vocab_size)!!;
    defer mem::free(first);

    test::eq(33036, (int)first.len);

    float lowest = 1.0e30f;
    float highest = -1.0e30f;
    foreach (value : first)
    {
        assert(value == value, "NaN in the logits");
        if (value < lowest) lowest = value;
        if (value > highest) highest = value;
    }
    assert(highest - lowest > 1.0f, "logits are flat — the graph produced nothing");
    assert(highest < 1.0e4f && lowest > -1.0e4f, "logits ran away");

    io::printfn("  skintokens transformer: 28 layers, logits in [%.2f, %.2f]", lowest, highest);

    // A different token has to give a different answer. Q8_0 weights and a
    // silently mis-wired graph can both still produce finite numbers; only
    // input sensitivity says the weights are being read as weights.
    model.forward(300, 0)!!;
    float[] second = download_tensor(mem, &ctx, &model.acts.logits, config.vocab_size)!!;
    defer mem::free(second);

    usz differing = 0;
    foreach (i, value : second)
    {
        if (value != first[i]) differing++;
    }
    assert(differing > first.len / 2, "two different tokens gave near-identical logits");

    // And the same token twice has to give the same answer.
    model.forward(1, 0)!!;
    float[] again = download_tensor(mem, &ctx, &model.acts.logits, config.vocab_size)!!;
    defer mem::free(again);

    foreach (i, value : again)
    {
        assert(value == first[i], "the same token gave different logits on a re-run");
    }
}
