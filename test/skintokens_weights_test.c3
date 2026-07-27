module llm;

import std::io;
import std::io::file;
import std::core::test;

// The converted SkinTokens weights, produced by:
//   python3 tools/convert_ckpt.py grpo_1400.ckpt --split --drop-training -o <dir>
// They are ~1.1 GB, so they live outside the repo; symlink or copy them here to
// run these. Absent, the tests report a skip rather than failing the suite.
const String WEIGHTS_DIR = "test/models/skintokens";

fn bool have(String name)
{
    @pool()
    {
        String full = string::tformat("%s/%s", WEIGHTS_DIR, name);
        if (try File f = file::open(full, "rb"))
        {
            (void)f.close();
            return true;
        }
        io::printfn("  skipped: %s not present", full);
        return false;
    };
}

fn void expect_shape(SafetensorsFile* sf, String name, ulong rows, ulong cols)
{
    SafetensorsTensorInfo* t = sf.find_tensor(name)!!;
    test::eq("BF16", t.dtype);
    test::eq(2, (int)t.n_dims);
    test::eq((int)rows, (int)t.shape[0]);
    test::eq((int)cols, (int)t.shape[1]);
    test::eq(rows * cols * 2, t.data_end - t.data_start);
}

// The shape encoder: 8 self-attention layers plus one cross-attention block,
// 512 wide. input_proj's 54 columns are the whole input contract — xyz, 48
// Fourier features (8 frequencies) and a 3-vector normal per sampled point.
fn void test_mesh_encoder_weights() @test
{
    if (!have("grpo_1400.mesh_encoder.safetensors")) return;

    SafetensorsFile sf = safetensors_open(string::tformat("%s/grpo_1400.mesh_encoder.safetensors", WEIGHTS_DIR))!!;
    defer sf.close();

    test::eq(109, (int)sf.n_tensors);

    expect_shape(&sf, "mesh_encoder.encoder.input_proj.weight", 512, 54);
    expect_shape(&sf, "mesh_encoder.encoder.cross_attn.attn.c_q.weight", 512, 512);
    expect_shape(&sf, "mesh_encoder.encoder.cross_attn.attn.c_kv.weight", 1024, 512);
    expect_shape(&sf, "mesh_encoder.encoder.self_attn.resblocks.7.attn.c_qkv.weight", 1536, 512);

    // The bridge into the language model, and the reason its width must match.
    expect_shape(&sf, "output_proj.0.weight", 896, 512);

    test::eq("512", sf.get_metadata("mesh_encoder_width")!!);
    test::eq("8", sf.get_metadata("mesh_encoder_num_encoder_layers")!!);
    test::eq("256", sf.get_metadata("mesh_encoder_num_latents")!!);
    test::eq("8", sf.get_metadata("mesh_encoder_num_freqs")!!);
    test::eq("False", sf.get_metadata("mesh_encoder_include_pi")!!);

    // No learned query parameters anywhere: the 256 latents are a farthest-point
    // subsample of the input cloud pushed through input_proj, not weights.
    test::@check(@catch(sf.find_tensor("mesh_encoder.encoder.learned_queries")) == llm::FILE_NOT_FOUND,
        "mesh encoder should have no learned_queries tensor");
}

// Qwen3's architecture at non-standard dimensions: hidden 896, and a vocabulary
// of skin tokens rather than text.
fn void test_transformer_weights() @test
{
    if (!have("grpo_1400.transformer.safetensors")) return;

    SafetensorsFile sf = safetensors_open(string::tformat("%s/grpo_1400.transformer.safetensors", WEIGHTS_DIR))!!;
    defer sf.close();

    test::eq(311, (int)sf.n_tensors);   // 28 layers * 11 + embed + norm + lm_head

    expect_shape(&sf, "transformer.model.embed_tokens.weight", 33036, 896);
    expect_shape(&sf, "transformer.lm_head.weight", 33036, 896);

    // Grouped-query attention: 16 query heads, 8 key/value heads, head_dim 128.
    // q_dim (2048) differs from the model width (896), which llama.c3 already
    // handles for Qwen3.
    expect_shape(&sf, "transformer.model.layers.0.self_attn.q_proj.weight", 2048, 896);
    expect_shape(&sf, "transformer.model.layers.0.self_attn.k_proj.weight", 1024, 896);
    expect_shape(&sf, "transformer.model.layers.0.self_attn.o_proj.weight", 896, 2048);
    expect_shape(&sf, "transformer.model.layers.27.mlp.gate_proj.weight", 3072, 896);

    // Per-head RMSNorm over head_dim — the Qwen3 signature.
    SafetensorsTensorInfo* q_norm = sf.find_tensor("transformer.model.layers.0.self_attn.q_norm.weight")!!;
    test::eq(1, (int)q_norm.n_dims);
    test::eq(128, (int)q_norm.shape[0]);

    test::eq("896", sf.get_metadata("llm_hidden_size")!!);
    test::eq("3192", sf.get_metadata("llm_max_position_embeddings")!!);
}

// Detokenising: tokens become latents, and the decoder evaluates a scalar skin
// weight at any query point.
fn void test_vae_decode_weights() @test
{
    if (!have("grpo_1400.vae.safetensors")) return;

    SafetensorsFile sf = safetensors_open(string::tformat("%s/grpo_1400.vae.safetensors", WEIGHTS_DIR))!!;
    defer sf.close();

    test::eq(202, (int)sf.n_tensors);

    // FSQ carries no codebook: 5 dimensions at 8 levels each enumerate all
    // 32768 tokens, so only the projections in and out are learned.
    expect_shape(&sf, "vae.model.FSQ.project_out.weight", 512, 5);
    expect_shape(&sf, "vae.model.post_quant.weight", 768, 512);

    // Query points enter through the same 54-dimensional embedding the shape
    // encoder uses, and leave as one weight each.
    expect_shape(&sf, "vae.model.decoder.proj_query.weight", 768, 54);
    expect_shape(&sf, "vae.model.decoder.proj_out.weight", 1, 768);

    // Ten self-attention blocks then one cross-attention block.
    expect_shape(&sf, "vae.model.decoder.blocks.9.attn1.to_q.weight", 768, 768);
    expect_shape(&sf, "vae.model.decoder.blocks.10.attn2.to_q.weight", 768, 768);

    test::eq("[8, 8, 8, 8, 8]", sf.get_metadata("model_FSQ_dict_levels")!!);
    test::eq("4", sf.get_metadata("model_config_tokens_per_skin")!!);

    // The encode path is training-only and was dropped by the converter.
    test::@check(@catch(sf.find_tensor("vae.model.encoder.learned_queries")) == llm::FILE_NOT_FOUND,
        "training-only VAE encoder should not be present");
    test::@check(@catch(sf.find_tensor("vae.model.FSQ.project_in.weight")) == llm::FILE_NOT_FOUND,
        "training-only FSQ project_in should not be present");
}
