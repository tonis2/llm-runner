module llm;

import llm::mesh;
import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::core::test;
import std::math;

// The batched prefill path had no callers in this engine before conditioning
// on a mesh needed it — it was written and never run. A disagreement with the
// single-token path would not crash or look wrong; the model would just
// produce plausible nonsense, which is the hardest kind of bug to see from the
// output. So the two paths are pinned against each other.
fn void test_prefill_matches_stepwise() @test
{
    String gguf = "test/models/skintokens/grpo_1400.gguf";
    if (try File f = file::open(gguf, "rb")) { (void)f.close(); } else { io::printfn("skip"); return; }

    DeviceContext ctx = createContext()!!;
    defer ctx.free();
    mmap::FileMmap w = file::mmap_open(gguf, "rb")!!;
    defer (void)w.destroy();
    GGUFFile gf = gguf_parse(w.bytes())!!;
    defer gf.free();
    String schema = (String)&llm_text::QWEN3_CONFIG_JSON;
    ModelConfig config = load_arch_config(schema, &gf)!!;
    defer config.free();
    WeightNames names = load_weight_names(schema)!!;
    defer names.free();

    llm_text::LlmModel model = llm_text::load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    uint[6] seq = { 257, 263, 100, 120, 140, 55 };
    uint dim = config.dim;

    // Reference: one token at a time through the path the engine already uses.
    for (uint i = 0; i < seq.len; i++) model.forward(seq[i], i)!!;
    float[] stepwise = download_tensor(mem, &ctx, &model.acts.logits, config.vocab_size)!!;
    defer mem::free(stepwise);

    // Same tokens, but as embeddings through the batched path.
    Tensor hidden = create_f32_tensor(&ctx, { (ulong)((usz)seq.len * dim), 0, 0, 0 }, 1)!!;
    defer hidden.free();

    ulong row = (ulong)dim * 4;
    RenderGraph g;
    g.init(&ctx)!!;
    for (uint i = 0; i < seq.len; i++)
    {
        g.copy_region(&model.weights.token_embedding, (usz)((ulong)seq[i] * row),
            &hidden, (usz)((ulong)i * row), row);
    }
    g.flush()!!;

    model.prefill_embeddings(&hidden, seq.len)!!;
    float[] batched = download_tensor(mem, &ctx, &model.acts.logits, config.vocab_size)!!;
    defer mem::free(batched);

    float worst = 0;
    usz worst_at = 0;
    for (usz i = 0; i < stepwise.len; i++)
    {
        float d = math::abs(stepwise[i] - batched[i]);
        if (d > worst) { worst = d; worst_at = i; }
    }

    float span = 0;
    float lo = stepwise[0];
    float hi = stepwise[0];
    foreach (v : stepwise) { if (v < lo) lo = v; if (v > hi) hi = v; }
    span = hi - lo;

    io::printfn("  stepwise vs prefill: max delta %.5f at %d, logit span %.3f", worst, worst_at, span);

    // Two argmaxes are the real question: do they pick the same next token?
    uint a = 0;
    uint b = 0;
    for (usz i = 1; i < stepwise.len; i++)
    {
        if (stepwise[i] > stepwise[a]) a = (uint)i;
        if (batched[i] > batched[b]) b = (uint)i;
    }
    io::printfn("  argmax stepwise %d, prefill %d", a, b);

    // Q8_0 weights and a different reduction order put a floor on how close
    // these can be; a real disagreement would be orders of magnitude larger
    // than a percent of the logit span.
    assert(worst < span * 0.02f,
        "prefill and stepwise disagree by %f against a span of %f", worst, span);
    assert(a == b, "the two paths would pick different next tokens (%d vs %d)", a, b);
}
