module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::test;
import llm_text;

// The converted SkinTokens checkpoint, produced by:
//   python3 tools/convert_ckpt.py grpo_1400.ckpt --gguf --drop-training \
//       --hparams-from last.ckpt -o grpo_1400.gguf
// One file holds all three networks. It is ~1 GB, so it lives outside the repo;
// symlink or copy it here to run these. Absent, the tests report a skip.
const String WEIGHTS = "test/models/skintokens/grpo_1400.gguf";

fn bool have()
{
    if (try File f = file::open(WEIGHTS, "rb"))
    {
        (void)f.close();
        return true;
    }
    io::printfn("  skipped: %s not present", WEIGHTS);
    return false;
}

// GGUF stores dimensions fastest-varying first, so a PyTorch [out, in] weight
// is described as ne = [in, out]. The bytes are the same either way; only the
// description is reversed, which is what the engine's loaders expect.
fn void expect_shape(GGUFFile* gf, String name, ulong out_dim, ulong in_dim, GGMLType type = GGML_F32)
{
    GGUFTensorInfo* t = gf.find_tensor(name)!!;
    test::eq((int)type, (int)t.type);
    test::eq(2, (int)t.n_dims);
    test::eq((int)in_dim, (int)t.shape[0]);
    test::eq((int)out_dim, (int)t.shape[1]);
}

fn void test_gguf_architecture_metadata() @test
{
    if (!have()) return;

    mmap::FileMmap weights = file::mmap_open(WEIGHTS, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    test::eq(622, (int)gf.n_tensors);

    // Written so the engine's own qwen3.json finds what it looks for. head_dim
    // is 128 while dim/heads is 56, which is exactly why the config reads
    // attention.key_length rather than deriving it.
    test::eq("qwen3", detect_architecture(&gf));

    ModelConfig config = load_arch_config((String)&llm_text::QWEN3_CONFIG_JSON, &gf)!!;
    defer config.free();

    test::eq(896, (int)config.dim);
    test::eq(28, (int)config.n_layers);
    test::eq(16, (int)config.n_heads);
    test::eq(8, (int)config.n_kv_heads);
    test::eq(128, (int)config.head_dim);
    test::eq(3072, (int)config.ffn_dim);
    test::eq(33036, (int)config.vocab_size);
}

// The shape encoder: 8 self-attention layers plus one cross-attention block,
// 512 wide. input_proj's 54 columns are the whole input contract — xyz, 48
// Fourier features (8 frequencies) and a 3-vector normal per sampled point.
fn void test_mesh_encoder_weights() @test
{
    if (!have()) return;

    mmap::FileMmap weights = file::mmap_open(WEIGHTS, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    expect_shape(&gf, "mesh_encoder.encoder.input_proj.weight", 512, 54);
    expect_shape(&gf, "mesh_encoder.encoder.cross_attn.attn.c_q.weight", 512, 512);
    expect_shape(&gf, "mesh_encoder.encoder.cross_attn.attn.c_kv.weight", 1024, 512);
    expect_shape(&gf, "mesh_encoder.encoder.self_attn.resblocks.7.attn.c_qkv.weight", 1536, 512);

    // The bridge into the language model, and the reason its width must match.
    expect_shape(&gf, "output_proj.0.weight", 896, 512);

    // No learned query parameters anywhere: the 512 latents are a farthest-point
    // subsample of the input cloud pushed through input_proj, not weights.
    test::@check(@catch(gf.find_tensor("mesh_encoder.encoder.learned_queries")) == llm::FILE_NOT_FOUND,
        "mesh encoder should have no learned_queries tensor");
}

// Qwen3's architecture at non-standard dimensions: hidden 896, and a vocabulary
// of skin tokens rather than text. Renamed to llama.cpp's scheme by the
// converter, so the engine's existing loader reads it without a special case.
fn void test_transformer_weights() @test
{
    if (!have()) return;

    mmap::FileMmap weights = file::mmap_open(WEIGHTS, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    // 2D weights are quantised; norms stay F32, where quantisation would cost
    // accuracy for a rounding error's worth of space.
    expect_shape(&gf, "token_embd.weight", 33036, 896, GGML_Q8_0);
    expect_shape(&gf, "output.weight", 33036, 896, GGML_Q8_0);

    // Grouped-query attention: 16 query heads, 8 key/value heads, head_dim 128.
    // q_dim (2048) differs from the model width (896), which llama.c3 already
    // handles for Qwen3.
    expect_shape(&gf, "blk.0.attn_q.weight", 2048, 896, GGML_Q8_0);
    expect_shape(&gf, "blk.0.attn_k.weight", 1024, 896, GGML_Q8_0);
    expect_shape(&gf, "blk.0.attn_output.weight", 896, 2048, GGML_Q8_0);
    expect_shape(&gf, "blk.27.ffn_gate.weight", 3072, 896, GGML_Q8_0);

    // Per-head RMSNorm over head_dim — the Qwen3 signature.
    GGUFTensorInfo* q_norm = gf.find_tensor("blk.0.attn_q_norm.weight")!!;
    test::eq(1, (int)q_norm.n_dims);
    test::eq(128, (int)q_norm.shape[0]);
    test::eq((int)GGML_F32, (int)q_norm.type);

    GGUFTensorInfo* attn_norm = gf.find_tensor("blk.0.attn_norm.weight")!!;
    test::eq((int)GGML_F32, (int)attn_norm.type);
}

// Detokenising: tokens become latents, and the decoder evaluates a scalar skin
// weight at any query point.
fn void test_vae_decode_weights() @test
{
    if (!have()) return;

    mmap::FileMmap weights = file::mmap_open(WEIGHTS, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    // FSQ carries no codebook: 5 dimensions at 8 levels each enumerate all
    // 32768 tokens, so only the projections in and out are learned.
    expect_shape(&gf, "vae.model.FSQ.project_out.weight", 512, 5);
    expect_shape(&gf, "vae.model.post_quant.weight", 768, 512);

    // Query points enter through the same 54-dimensional embedding the shape
    // encoder uses, and leave as one weight each.
    expect_shape(&gf, "vae.model.decoder.proj_query.weight", 768, 54);
    expect_shape(&gf, "vae.model.decoder.proj_out.weight", 1, 768);

    // Ten self-attention blocks then one cross-attention block.
    expect_shape(&gf, "vae.model.decoder.blocks.9.attn1.to_q.weight", 768, 768);
    expect_shape(&gf, "vae.model.decoder.blocks.10.attn2.to_q.weight", 768, 768);

    // The encode path is training-only and was dropped by the converter.
    test::@check(@catch(gf.find_tensor("vae.model.encoder.norm_out.weight")) == llm::FILE_NOT_FOUND,
        "training-only VAE encoder should not be present");
    test::@check(@catch(gf.find_tensor("vae.model.FSQ.project_in.weight")) == llm::FILE_NOT_FOUND,
        "training-only FSQ project_in should not be present");
}

// The hyperparameters travel with the weights, so one file is enough to
// rebuild the pipeline.
fn void test_gguf_carries_hyperparameters() @test
{
    if (!have()) return;

    mmap::FileMmap weights = file::mmap_open(WEIGHTS, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    test::eq("512", gf.get_string("skintokens.mesh_encoder_width")!!);
    // token_num, not num_latents: 512 queries, and the 256 is a learned-query
    // count this encoder never uses.
    test::eq("512", gf.get_string("skintokens.mesh_encoder_token_num")!!);
    test::eq("8", gf.get_string("skintokens.mesh_encoder_num_freqs")!!);
    test::eq("False", gf.get_string("skintokens.mesh_encoder_include_pi")!!);
    test::eq("[8, 8, 8, 8, 8]", gf.get_string("skintokens.model_FSQ_dict_levels")!!);
    test::eq("4", gf.get_string("skintokens.model_config_tokens_per_skin")!!);
    test::eq("3192", gf.get_string("skintokens.llm_n_positions")!!);
}
