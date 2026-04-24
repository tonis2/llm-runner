module vit_test;

import vk;
import llm;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::math;
import std::core::mem;

const char[*] LLM_SPV = $embed("../../dependencies/llm_text.c3l/shaders/llm.spv");

struct BatchMatMulParams { uint out_dim; uint in_dim; uint seq_len; uint row_offset; }

fn void? inspect_gguf_dtypes(String path) {
    io::printfn("\n=== Inspecting GGUF: %s ===", path);
    
    mmap::FileMmap mm = file::mmap_open(path, "rb")!!;
    char[] data = mm.bytes();
    
    llm::GGUFReader reader = { .data = data, .pos = 0 };
    
    // Read header
    uint magic = reader.read_u32();
    if (magic != 0x46554747) {
        io::printfn("  Not a valid GGUF file (magic=0x%08X)", magic);
        return;
    }
    uint version = reader.read_u32();
    uint n_tensors = reader.read_u32();
    uint n_kv = reader.read_u32();
    
    io::printfn("  Version: %d, Tensors: %d, KV pairs: %d", version, n_tensors, n_kv);
    
    // Skip metadata
    for (uint i = 0; i < n_kv; i++) {
        String key = reader.read_string();
        uint val_type = reader.read_u32();
        reader.skip_value((llm::GGUFType)val_type);
    }
    
    // Read tensor info
    uint q4_0_count = 0;
    uint q8_0_count = 0;
    uint f32_count = 0;
    uint other_count = 0;
    
    for (uint i = 0; i < n_tensors; i++) {
        String name = reader.read_string();
        uint n_dims = reader.read_u32();
        ulong[4] shape;
        for (uint d = 0; d < n_dims; d++) shape[d] = reader.read_u64();
        uint dtype = reader.read_u32();
        ulong offset = reader.read_u64();
        
        switch (dtype) {
            case 2:  // GGML_Q4_0
                q4_0_count++;
            case 8:  // GGML_Q8_0
                q8_0_count++;
            case 0:  // GGML_F32
                f32_count++;
            default:
                other_count++;
                io::printfn("  %-40s dtype=%d shape=[%d,%d,%d,%d]", name, dtype, 
                    (uint)shape[0], (uint)shape[1], (uint)shape[2], (uint)shape[3]);
        }
    }
    
    io::printfn("  Summary: Q4_0=%d, Q8_0=%d, F32=%d, Other=%d", q4_0_count, q8_0_count, f32_count, other_count);
}

fn void? test_q4_0_matmul(llm::DeviceContext* ctx) {
    io::printfn("\n=== Q4_0 Matmul Test ===");

    uint in_dim = 3072;
    uint out_dim = 3072;
    uint seq_len = 256;

    uint blocks_per_row = in_dim / 32;
    uint bytes_per_row = blocks_per_row * 18;
    uint total_bytes = out_dim * bytes_per_row;

    // Create Q4_0 weights: all 1.0f
    char[] weight_data = mem::new_array(char, total_bytes);
    defer mem::free(weight_data);

    for (uint row = 0; row < out_dim; row++) {
        for (uint block = 0; block < blocks_per_row; block++) {
            uint block_offset = row * bytes_per_row + block * 18;
            llm::cpu_f32_to_f16(&weight_data[block_offset], 0.5f);
            // nibble = round(1.0f / 0.5f) + 8 = 10
            for (uint byte_idx = 0; byte_idx < 16; byte_idx++) {
                uint nibble = 10;
                weight_data[block_offset + 2 + byte_idx] = (char)((nibble << 4) | nibble);
            }
        }
    }

    llm::Tensor weight = {
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: weight_data.ptr,
            data_size: total_bytes,
        )!!,
        .dtype = llm::GGML_Q4_0,
        .n_dims = 2,
        .shape = { in_dim, out_dim, 0, 0 },
        .size_bytes = total_bytes,
    };
    defer weight.free();

    // Input: all 1.0f
    float[] input_data = mem::new_array(float, in_dim);
    defer mem::free(input_data);
    for (uint i = 0; i < in_dim; i++) input_data[i] = 1.0f;

    llm::Tensor input = {
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: input_data.ptr,
            data_size: (usz)(in_dim * 4),
        )!!,
        .dtype = llm::GGML_F32,
        .n_dims = 1,
        .shape = { in_dim, 0, 0, 0 },
        .size_bytes = (usz)(in_dim * 4),
    };
    defer input.free();

    // Output buffer
    usz output_bytes = (usz)(seq_len * out_dim * 4);
    llm::Tensor output = llm::create_f32_tensor(ctx, { out_dim, 0, 0, 0 }, 1)!!;
    defer output.free();

    // Create kernel
    char[] llm_spv = &LLM_SPV;
    vk::ShaderModule shader = vk::shaderModuleCreateInfo().setCodeSize(
        llm_spv.len
    ).setCode((uint*)&llm_spv[0]).build(ctx.device)!!;

    llm::ComputeKernel kernel = llm::create_kernel(ctx, shader, 3, BatchMatMulParams.sizeof, "batch_matmul_q4_0")!!;
    defer kernel.free(ctx.device);

    // Dispatch
    BatchMatMulParams pc = { .out_dim = out_dim, .in_dim = in_dim, .seq_len = seq_len, .row_offset = 0 };
    llm::RenderGraph g;
    g.init(ctx)!!;
    g.dispatch(&kernel, { &weight, &input, &output }, &pc, seq_len * out_dim);
    g.flush()!!;

    // Download and verify
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: output_bytes,
    )!!;
    defer staging.free();

    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, output.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = output_bytes }});
    }!!;

    float* mapped = (float*)staging.data();

    // CPU reference: dequantize and matmul
    float[] expected = mem::new_array(float, out_dim);
    defer mem::free(expected);

    for (uint row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        for (uint block = 0; block < blocks_per_row; block++) {
            uint block_offset = row * bytes_per_row + block * 18;
            float scale = llm::cpu_f16_to_f32(&weight_data[block_offset]);
            for (uint byte_idx = 0; byte_idx < 16; byte_idx++) {
                char byte_val = weight_data[block_offset + 2 + byte_idx];
                int nibble_lo = (byte_val & 0xF) - 8;
                int nibble_hi = ((byte_val >> 4) & 0xF) - 8;
                uint base = block * 32 + byte_idx;
                sum += scale * nibble_lo * input_data[base];
                sum += scale * nibble_hi * input_data[base + 16];
            }
        }
        expected[row] = sum;
    }

    uint errors = 0;
    for (uint i = 0; i < out_dim; i++) {
        float diff = mapped[i] > expected[i] ? mapped[i] - expected[i] : expected[i] - mapped[i];
        if (diff > 0.01f) {
            if (errors < 5) {
                io::printfn("  MISMATCH @ %d: GPU=%.4f CPU=%.4f diff=%.4f", i, mapped[i], expected[i], diff);
            }
            errors++;
        }
    }

    if (errors == 0) {
        io::printfn("  PASS: All %d outputs match (expected ~%.1f)", out_dim, expected[0]);
    } else {
        io::printfn("  FAIL: %d/%d outputs mismatch", errors, out_dim);
        io::printfn("  First 10 GPU: [");
        for (uint i = 0; i < 10; i++) {
            if (i > 0) io::printf(", ");
            io::printf("%.4f", mapped[i]);
        }
        io::printfn("]");
        io::printfn("  First 10 CPU: [");
        for (uint i = 0; i < 10; i++) {
            if (i > 0) io::printf(", ");
            io::printf("%.4f", expected[i]);
        }
        io::printfn("]");
    }
}
