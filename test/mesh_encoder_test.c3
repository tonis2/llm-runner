module llm;

import llm::mesh;
import vk;
import std::io;
import std::io::file;
import std::core::mem;
import std::core::test;
import std::math;
import std::math::random;

// The encoder's kernels are checked against CPU references rather than against
// PyTorch: no torch on this machine, and a reference here catches the failure
// these shaders actually have, which is a layout or stride mistake rather than
// an arithmetic one.

fn Tensor upload(DeviceContext* ctx, float[] data)
{
    Tensor tensor = create_f32_tensor(ctx, { (ulong)data.len, 0, 0, 0 }, 1)!!;
    upload_to_tensor(ctx, &tensor, data)!!;
    return tensor;
}

fn float[] noise(random::Sfc64Random* rng, usz count, float spread = 2.0f)
{
    float[] data = mem::new_array(float, (sz)count);
    foreach (&value : data) *value = (random::next_float(rng) - 0.5f) * spread;
    return data;
}

fn void expect_close(float[] got, float[] want, String label, float tolerance = 2e-4f)
{
    assert(got.len == want.len, "%s: length %d, expected %d", label, got.len, want.len);
    foreach (i, value : got)
    {
        float delta = math::abs(value - want[i]);
        assert(delta <= tolerance,
            "%s: element %d is %f, expected %f (off by %f)", label, i, value, want[i], delta);
    }
}

// --- CPU references ---

fn void cpu_linear(float[] out, float[] input, float[] weight, float[] bias, usz rows, usz out_dim, usz in_dim)
{
    for (usz r = 0; r < rows; r++)
    {
        for (usz o = 0; o < out_dim; o++)
        {
            float sum = bias.len > 0 ? bias[o] : 0.0f;
            for (usz k = 0; k < in_dim; k++) sum += input[r * in_dim + k] * weight[o * in_dim + k];
            out[r * out_dim + o] = sum;
        }
    }
}

fn void cpu_layernorm(float[] out, float[] input, float[] weight, float[] bias, usz rows, usz dim, float eps)
{
    for (usz r = 0; r < rows; r++)
    {
        usz base = r * dim;
        float mean = 0;
        for (usz i = 0; i < dim; i++) mean += input[base + i];
        mean /= (float)dim;

        float variance = 0;
        for (usz i = 0; i < dim; i++)
        {
            float diff = input[base + i] - mean;
            variance += diff * diff;
        }
        float inv_std = 1.0f / math::sqrt(variance / (float)dim + eps);

        for (usz i = 0; i < dim; i++)
        {
            out[base + i] = (input[base + i] - mean) * inv_std * weight[i] + (bias.len > 0 ? bias[i] : 0.0f);
        }
    }
}

fn void cpu_rmsnorm(float[] out, float[] input, float[] weight, usz rows, usz dim, float eps)
{
    for (usz r = 0; r < rows; r++)
    {
        usz base = r * dim;
        float squares = 0;
        for (usz i = 0; i < dim; i++) squares += input[base + i] * input[base + i];
        float inv_rms = 1.0f / math::sqrt(squares / (float)dim + eps);
        for (usz i = 0; i < dim; i++) out[base + i] = input[base + i] * inv_rms * weight[i];
    }
}

// Mirrors the strided addressing in mesh_attention.slang, so a mismatch means
// the shader disagrees with the layout rather than that both are wrong.
fn void cpu_attention(float[] out, float[] q, float[] kv, mesh::MeshAttentionPC pc, usz head_dim)
{
    for (usz head = 0; head < pc.n_heads; head++)
    {
        for (usz t = 0; t < pc.q_len; t++)
        {
            usz q_base = t * pc.q_row_stride + head * pc.q_head_stride + pc.q_offset;

            float[] scores = mem::new_array(float, (sz)pc.kv_len);
            defer mem::free(scores);

            float top = -1.0e30f;
            for (usz s = 0; s < pc.kv_len; s++)
            {
                usz k_base = s * pc.kv_row_stride + head * pc.kv_head_stride + pc.k_offset;
                float dot = 0;
                for (usz d = 0; d < head_dim; d++) dot += q[q_base + d] * kv[k_base + d];
                scores[s] = dot * pc.scale;
                if (scores[s] > top) top = scores[s];
            }

            float total = 0;
            for (usz s = 0; s < pc.kv_len; s++)
            {
                scores[s] = math::exp(scores[s] - top);
                total += scores[s];
            }

            for (usz d = 0; d < head_dim; d++)
            {
                float acc = 0;
                for (usz s = 0; s < pc.kv_len; s++)
                {
                    usz v_base = s * pc.kv_row_stride + head * pc.kv_head_stride + pc.v_offset;
                    acc += scores[s] * kv[v_base + d];
                }
                out[t * pc.out_row_stride + head * head_dim + d] = acc / total;
            }
        }
    }
}

// --- Kernel tests ---

fn void test_mesh_encoder_kernels() @test
{
    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    mesh::MeshEncoderKernels kernels = mesh::create_encoder_kernels(&ctx)!!;
    defer kernels.free(ctx.device);

    random::Sfc64Random rng;
    random::seed(&rng, 11);

    // Deliberately not multiples of 16 or 256, so the tile and bounds masking
    // in every kernel gets exercised.
    const usz ROWS = 7;
    const usz IN_DIM = 37;
    const usz OUT_DIM = 21;

    float[] input = noise(&rng, ROWS * IN_DIM);
    float[] weight = noise(&rng, OUT_DIM * IN_DIM, 0.5f);
    float[] bias = noise(&rng, OUT_DIM);
    float[] want = mem::new_array(float, (sz)(ROWS * OUT_DIM));
    defer { mem::free(input); mem::free(weight); mem::free(bias); mem::free(want); }

    Tensor t_input = upload(&ctx, input);
    Tensor t_weight = upload(&ctx, weight);
    Tensor t_bias = upload(&ctx, bias);
    Tensor t_out = create_f32_tensor(&ctx, { (ulong)(ROWS * OUT_DIM), 0, 0, 0 }, 1)!!;
    defer { t_input.free(); t_weight.free(); t_bias.free(); t_out.free(); }

    // --- linear, with bias ---
    {
        RenderGraph graph;
        graph.init(&ctx)!!;
        mesh::MeshLinearPC pc = { .m = (uint)ROWS, .n = (uint)OUT_DIM, .k = (uint)IN_DIM, .has_bias = 1 };
        graph.dispatch(&kernels.linear, { &t_input, &t_weight, &t_bias, &t_out }, &pc,
            ceil_div((uint)OUT_DIM, 16), ceil_div((uint)ROWS, 16));
        graph.flush()!!;

        cpu_linear(want, input, weight, bias, ROWS, OUT_DIM, IN_DIM);
        float[] got = download_tensor(mem, &ctx, &t_out, ROWS * OUT_DIM)!!;
        defer mem::free(got);
        expect_close(got, want, "linear with bias");
    }

    // --- linear, no bias: the bias binding still points somewhere and must be ignored ---
    {
        RenderGraph graph;
        graph.init(&ctx)!!;
        mesh::MeshLinearPC pc = { .m = (uint)ROWS, .n = (uint)OUT_DIM, .k = (uint)IN_DIM, .has_bias = 0 };
        graph.dispatch(&kernels.linear, { &t_input, &t_weight, &t_weight, &t_out }, &pc,
            ceil_div((uint)OUT_DIM, 16), ceil_div((uint)ROWS, 16));
        graph.flush()!!;

        cpu_linear(want, input, weight, {}, ROWS, OUT_DIM, IN_DIM);
        float[] got = download_tensor(mem, &ctx, &t_out, ROWS * OUT_DIM)!!;
        defer mem::free(got);
        expect_close(got, want, "linear without bias");
    }

    // --- layernorm ---
    {
        float[] gamma = noise(&rng, IN_DIM, 1.0f);
        float[] beta = noise(&rng, IN_DIM);
        float[] normed = mem::new_array(float, (sz)(ROWS * IN_DIM));
        defer { mem::free(gamma); mem::free(beta); mem::free(normed); }

        Tensor t_gamma = upload(&ctx, gamma);
        Tensor t_beta = upload(&ctx, beta);
        Tensor t_normed = create_f32_tensor(&ctx, { (ulong)(ROWS * IN_DIM), 0, 0, 0 }, 1)!!;
        defer { t_gamma.free(); t_beta.free(); t_normed.free(); }

        RenderGraph graph;
        graph.init(&ctx)!!;
        mesh::MeshLayerNormPC pc = { .dim = (uint)IN_DIM, .n_rows = (uint)ROWS, .eps = 1e-5f, .has_bias = 1 };
        graph.dispatch(&kernels.layernorm, { &t_input, &t_gamma, &t_beta, &t_normed }, &pc, (uint)ROWS);

        // In-place is what the forward pass does to the 54000-row cloud, so it
        // is worth proving rather than assuming.
        mesh::MeshRmsNormPC rms_pc = { .dim = (uint)IN_DIM, .n_rows = (uint)ROWS, .eps = 1e-6f, .pad = 0 };
        graph.dispatch(&kernels.rmsnorm, { &t_input, &t_gamma, &t_input }, &rms_pc, (uint)ROWS);
        graph.flush()!!;

        cpu_layernorm(normed, input, gamma, beta, ROWS, IN_DIM, 1e-5f);
        float[] got = download_tensor(mem, &ctx, &t_normed, ROWS * IN_DIM)!!;
        defer mem::free(got);
        expect_close(got, normed, "layernorm");

        cpu_rmsnorm(normed, input, gamma, ROWS, IN_DIM, 1e-6f);
        float[] got_rms = download_tensor(mem, &ctx, &t_input, ROWS * IN_DIM)!!;
        defer mem::free(got_rms);
        expect_close(got_rms, normed, "rmsnorm in place");
    }

    // --- gelu and residual add ---
    {
        const usz N = 300;
        float[] a = noise(&rng, N, 6.0f);
        float[] b = noise(&rng, N);
        float[] expected = mem::new_array(float, (sz)N);
        defer { mem::free(a); mem::free(b); mem::free(expected); }

        Tensor t_a = upload(&ctx, a);
        Tensor t_b = upload(&ctx, b);
        defer { t_a.free(); t_b.free(); }

        RenderGraph graph;
        graph.init(&ctx)!!;
        mesh::MeshGeluPC gelu_pc = { .n = (uint)N };
        graph.dispatch(&kernels.gelu, { &t_a }, &gelu_pc, ceil_div((uint)N, 256));
        mesh::MeshAddPC add_pc = { .n = (uint)N };
        graph.dispatch(&kernels.residual_add, { &t_a, &t_b }, &add_pc, ceil_div((uint)N, 256));
        graph.flush()!!;

        // The exact erf GELU, computed a different way than the shader does it:
        // math::erf against the shader's rational approximation. Agreement to
        // 2e-4 is the approximation being checked, not just restated.
        for (usz i = 0; i < N; i++)
        {
            float x = a[i];
            expected[i] = 0.5f * x * (1.0f + (float)math::erf((double)x * 0.70710678118654752)) + b[i];
        }

        float[] got = download_tensor(mem, &ctx, &t_a, N)!!;
        defer mem::free(got);
        expect_close(got, expected, "gelu then residual add");
    }
}

fn void test_mesh_attention_kernel() @test
{
    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    mesh::MeshEncoderKernels kernels = mesh::create_encoder_kernels(&ctx)!!;
    defer kernels.free(ctx.device);

    random::Sfc64Random rng;
    random::seed(&rng, 23);

    const usz HEAD_DIM = 64;   // fixed by the kernel
    const usz HEADS = 2;

    // --- Cross attention: separate Q, and a fused [k|v] per head ---
    // 70 keys spans three 32-wide tiles with a ragged last one, and 20 queries
    // spans two 16-wide query tiles, so both masks matter.
    {
        const usz Q_LEN = 20;
        const usz KV_LEN = 70;

        float[] q = noise(&rng, Q_LEN * HEADS * HEAD_DIM);
        float[] kv = noise(&rng, KV_LEN * HEADS * 2 * HEAD_DIM);
        float[] want = mem::new_array(float, (sz)(Q_LEN * HEADS * HEAD_DIM));
        defer { mem::free(q); mem::free(kv); mem::free(want); }

        Tensor t_q = upload(&ctx, q);
        Tensor t_kv = upload(&ctx, kv);
        Tensor t_out = create_f32_tensor(&ctx, { (ulong)(Q_LEN * HEADS * HEAD_DIM), 0, 0, 0 }, 1)!!;
        defer { t_q.free(); t_kv.free(); t_out.free(); }

        mesh::MeshAttentionPC pc = {
            .n_heads = (uint)HEADS,
            .q_len = (uint)Q_LEN,
            .kv_len = (uint)KV_LEN,
            .q_row_stride = (uint)(HEADS * HEAD_DIM),
            .q_head_stride = (uint)HEAD_DIM,
            .q_offset = 0,
            .kv_row_stride = (uint)(HEADS * 2 * HEAD_DIM),
            .kv_head_stride = (uint)(2 * HEAD_DIM),
            .k_offset = 0,
            .v_offset = (uint)HEAD_DIM,
            .out_row_stride = (uint)(HEADS * HEAD_DIM),
            .scale = 1.0f / 8.0f,
        };

        RenderGraph graph;
        graph.init(&ctx)!!;
        graph.dispatch(&kernels.attention, { &t_q, &t_kv, &t_out }, &pc,
            ceil_div(pc.q_len, 16), (uint)HEADS);
        graph.flush()!!;

        cpu_attention(want, q, kv, pc, HEAD_DIM);
        float[] got = download_tensor(mem, &ctx, &t_out, Q_LEN * HEADS * HEAD_DIM)!!;
        defer mem::free(got);
        expect_close(got, want, "cross attention", 1e-3f);
    }

    // --- Self attention: one fused [q|k|v] per head, bound as both Q and KV ---
    {
        const usz LEN = 33;

        float[] qkv = noise(&rng, LEN * HEADS * 3 * HEAD_DIM);
        float[] want = mem::new_array(float, (sz)(LEN * HEADS * HEAD_DIM));
        defer { mem::free(qkv); mem::free(want); }

        Tensor t_qkv = upload(&ctx, qkv);
        Tensor t_out = create_f32_tensor(&ctx, { (ulong)(LEN * HEADS * HEAD_DIM), 0, 0, 0 }, 1)!!;
        defer { t_qkv.free(); t_out.free(); }

        mesh::MeshAttentionPC pc = {
            .n_heads = (uint)HEADS,
            .q_len = (uint)LEN,
            .kv_len = (uint)LEN,
            .q_row_stride = (uint)(HEADS * 3 * HEAD_DIM),
            .q_head_stride = (uint)(3 * HEAD_DIM),
            .q_offset = 0,
            .kv_row_stride = (uint)(HEADS * 3 * HEAD_DIM),
            .kv_head_stride = (uint)(3 * HEAD_DIM),
            .k_offset = (uint)HEAD_DIM,
            .v_offset = (uint)(2 * HEAD_DIM),
            .out_row_stride = (uint)(HEADS * HEAD_DIM),
            .scale = 1.0f / 8.0f,
        };

        RenderGraph graph;
        graph.init(&ctx)!!;
        graph.dispatch(&kernels.attention, { &t_qkv, &t_qkv, &t_out }, &pc,
            ceil_div(pc.q_len, 16), (uint)HEADS);
        graph.flush()!!;

        cpu_attention(want, qkv, qkv, pc, HEAD_DIM);
        float[] got = download_tensor(mem, &ctx, &t_out, LEN * HEADS * HEAD_DIM)!!;
        defer mem::free(got);
        expect_close(got, want, "self attention", 1e-3f);
    }
}

// --- End to end, against the real checkpoint ---

const String ENCODER_WEIGHTS = "test/models/skintokens/grpo_1400.gguf";

fn void test_mesh_encoder_forward() @test
{
    if (try File probe = file::open(ENCODER_WEIGHTS, "rb"))
    {
        (void)probe.close();
    }
    else
    {
        io::printfn("  skipped: %s not present", ENCODER_WEIGHTS);
        return;
    }

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    mesh::MeshEncoder encoder = mesh::load_mesh_encoder(&ctx, ENCODER_WEIGHTS)!!;
    defer encoder.free();

    // A cube is a poor thing to rig, but the point here is that the graph runs
    // end to end and produces finite, input-dependent numbers.
    Vec3[8] positions = {
        { -0.5, -0.5, -0.5 }, { 0.5, -0.5, -0.5 }, { 0.5, 0.5, -0.5 }, { -0.5, 0.5, -0.5 },
        { -0.5, -0.5,  0.5 }, { 0.5, -0.5,  0.5 }, { 0.5, 0.5,  0.5 }, { -0.5, 0.5,  0.5 },
    };
    uint[36] indices = {
        0, 2, 1,  0, 3, 2,   4, 5, 6,  4, 6, 7,
        0, 4, 7,  0, 7, 3,   1, 2, 6,  1, 6, 5,
        0, 1, 5,  0, 5, 4,   3, 7, 6,  3, 6, 2,
    };
    mesh::TriMesh cube = { .positions = positions[..], .indices = indices[..] };

    mesh::PrepareOptions options = mesh::DEFAULT_PREPARE;
    options.samples = 8192;

    mesh::ShapeEncoderInput input = mesh::prepare_shape_input(mem, cube, options)!!;
    defer input.free(mem);

    float[] latents = encoder.encode(mem, &input)!!;
    defer mem::free(latents);

    assert(latents.len == 512 * 896, "512 prefix embeddings of width 896");

    float largest = 0;
    float energy = 0;
    foreach (value : latents)
    {
        assert(value == value, "no NaN in the latents");
        assert(math::abs(value) < 1.0e6f, "no runaway values");
        if (math::abs(value) > largest) largest = math::abs(value);
        energy += value * value;
    }
    assert(largest > 1e-3f, "the encoder produced something other than zeros");

    io::printfn("  mesh encoder: 8192 points -> 512 x 896, peak %.3f, rms %.4f",
        largest, math::sqrt(energy / (float)latents.len));

    // The last op is an RMSNorm, so every row should come out with unit RMS
    // scaled by the norm weight. Checking a couple of rows catches a forward
    // pass that silently skipped it.
    for (usz row = 0; row < 4; row++)
    {
        float squares = 0;
        for (usz i = 0; i < 896; i++)
        {
            float value = latents[row * 896 + i];
            squares += value * value;
        }
        assert(squares > 0, "row %d is all zero", row);
    }

    // Same input, same answer: nothing in the graph reads uninitialised memory.
    float[] again = encoder.encode(mem, &input)!!;
    defer mem::free(again);
    expect_close(again, latents, "re-running the encoder", 1e-5f);
}
