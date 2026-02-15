module llm::pipelines;

import vk;
import std::io;
import std::io::file;
import std::core::mem;
import std::encoding::json;
import std::collections::object;
import image;
import image::png;
import llm;

// Embedded diffusion SPIR-V shader
const char[*] DIFFUSION_SPV = $embed("../shaders/diffusion.spv");

// Embedded SD config
const char[*] SD_CONFIG_JSON = $embed("../configs/sd.json");

// Fault definitions for pipeline errors
faultdef PIPELINE_DEVICE_ERROR,
         PIPELINE_GENERATION_FAILED,
         PIPELINE_INVALID_INPUT,
         PIPELINE_NOT_INITIALIZED;

// --- Pluggable Pipeline Interface ---
// Diffusers implement this interface. The engine delegates to the active impl.
// Types implementing this interface: ZImageState, SDXSState
interface Pipeline {
    fn void? load(llm::DeviceContext* ctx, String model_path, PipelineOptions* opts);
    fn image::Image? generate(llm::DeviceContext* ctx, GenerationInputs* inputs);
    fn void free();
}

struct PipelineOptions {
    String text_model_path;
    String vae_path;
    String taesd_path;
}

// --- Unified Generation Inputs ---

struct GenerationInputs {
    String prompt;
    uint image_size;
    uint num_steps;
    float cfg_scale;
    uint seed;

    // Optional: for img2img mode (if null, does txt2img)
    image::Image* input_image;
    float strength;          // 0.0-1.0, only used with input_image
}

fn GenerationInputs generation_inputs_defaults() {
    return {
        .prompt = "",
        .image_size = 512,
        .num_steps = 4,
        .cfg_scale = 7.0,
        .seed = 42,
        .input_image = null,
        .strength = 0.75,
    };
}

fn bool GenerationInputs.validate(GenerationInputs* self) {
    if (self.prompt.len == 0) {
        io::printfn("Error: prompt is required");
        return false;
    }
    if (self.num_steps == 0) {
        io::printfn("Error: num_steps must be > 0");
        return false;
    }
    if (self.cfg_scale < 1.0) {
        io::printfn("Error: cfg_scale must be >= 1.0");
        return false;
    }
    if (self.input_image != null && (self.strength < 0.0 || self.strength > 1.0)) {
        io::printfn("Error: strength must be in [0.0, 1.0]");
        return false;
    }
    return true;
}

// --- Diffusion Config ---

struct ClipConfig {
    uint dim;
    uint n_layers;
    uint n_heads;
    uint head_dim;
    uint max_tokens;
    uint vocab_size;
    float eps;
    String prefix;
}

struct UnetConfig {
    uint in_channels;
    uint out_channels;
    uint model_channels;
    uint[3] channel_mult;
    uint num_res_blocks;
    uint n_heads;
    uint context_dim;
    String prefix;
}

struct VaeConfig {
    uint latent_channels;
    uint out_channels;
    String prefix_decoder;
    String prefix_encoder;
}

struct ImageConfig {
    uint size;
    uint latent_size;
}

struct SchedulerConfig {
    uint num_train_timesteps;
    float beta_start;
    float beta_end;
}

struct DiffusionConfig {
    String name;
    ClipConfig clip;
    UnetConfig unet;
    VaeConfig vae;
    ImageConfig image;
    SchedulerConfig scheduler;
}

// --- Push constant structs matching Slang ---

struct Conv2dPC {
    uint in_c;
    uint out_c;
    uint in_h;
    uint in_w;
    uint kH;
    uint kW;
    uint stride;
    uint pad;
    uint groups;
    uint out_h;
    uint out_w;
    uint has_bias;
}

struct GroupNormPC {
    uint channels;
    uint spatial;
    uint num_groups;
    float eps;
}

struct BatchedMatMulPC {
    uint m_dim;
    uint n_dim;
    uint k_dim;
}

struct SpatialAttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
}

struct CrossAttentionPC {
    uint head_dim;
    uint n_heads;
    uint q_len;
    uint kv_len;
    float scale;
}

struct UpsamplePC {
    uint channels;
    uint in_h;
    uint in_w;
}

struct TimestepEmbedPC {
    uint dim;
    float timestep;
    float max_period;
}

struct BroadcastAddPC {
    uint channels;
    uint spatial;
}

struct ChannelConcatPC {
    uint channels_a;
    uint channels_b;
    uint spatial;
}

struct DdimStepPC {
    uint n;
    float sqrt_alpha_t;
    float sqrt_one_minus_alpha_t;
    float sqrt_alpha_prev;
    float sqrt_one_minus_alpha_prev;
}

struct EulerStepPC {
    uint n;
    float dt;
}

struct ScaleShiftPC {
    uint n;
    float scale;
    float shift;
}

struct F16ToF32PC {
    uint n;
}

struct ReluPC {
    uint n;
}

struct TanhClampPC {
    uint n;
}

// --- Diffusion Kernels ---

struct DiffusionKernels {
    llm::ComputeKernel conv2d;
    llm::ComputeKernel conv2d_q8;
    llm::ComputeKernel group_norm;
    llm::ComputeKernel batched_matmul;
    llm::ComputeKernel batched_matmul_q8;
    llm::ComputeKernel spatial_attention;
    llm::ComputeKernel cross_attention;
    llm::ComputeKernel upsample_nearest;
    llm::ComputeKernel timestep_embed;
    llm::ComputeKernel broadcast_add;
    llm::ComputeKernel channel_concat;
    llm::ComputeKernel ddim_step;
    llm::ComputeKernel euler_step;
    llm::ComputeKernel scale_shift_clamp;
    llm::ComputeKernel f16_to_f32;
    llm::ComputeKernel relu;
    llm::ComputeKernel tanh_clamp;
    llm::SharedKernels shared;
}

// --- Diffusion Model ---

struct DiffusionModel {
    DiffusionConfig config;
    DiffusionKernels kernels;
    llm::DeviceContext* ctx;
}

fn DiffusionConfig? load_diffusion_config(String json_text) {
    Object* root = json::parse_string(mem, json_text)!;

    String name = root.get_string("name") ?? "stable-diffusion";

    // CLIP config
    Object* clip_obj = root.get("clip")!;
    ClipConfig clip = {
        .dim = (uint)(clip_obj.get_float("dim") ?? 768.0),
        .n_layers = (uint)(clip_obj.get_float("n_layers") ?? 12.0),
        .n_heads = (uint)(clip_obj.get_float("n_heads") ?? 12.0),
        .head_dim = (uint)(clip_obj.get_float("head_dim") ?? 64.0),
        .max_tokens = (uint)(clip_obj.get_float("max_tokens") ?? 77.0),
        .vocab_size = (uint)(clip_obj.get_float("vocab_size") ?? 49408.0),
        .eps = (float)(clip_obj.get_float("eps") ?? 1e-5),
        .prefix = clip_obj.get_string("prefix") ?? "cond_stage_model.transformer.text_model.",
    };

    // UNet config
    Object* unet_obj = root.get("unet")!;
    UnetConfig unet = {
        .in_channels = (uint)(unet_obj.get_float("in_channels") ?? 4.0),
        .out_channels = (uint)(unet_obj.get_float("out_channels") ?? 4.0),
        .model_channels = (uint)(unet_obj.get_float("model_channels") ?? 320.0),
        .channel_mult = { 1, 2, 4 },
        .num_res_blocks = (uint)(unet_obj.get_float("num_res_blocks") ?? 2.0),
        .n_heads = (uint)(unet_obj.get_float("n_heads") ?? 8.0),
        .context_dim = (uint)(unet_obj.get_float("context_dim") ?? 768.0),
        .prefix = unet_obj.get_string("prefix") ?? "model.diffusion_model.",
    };

    // Parse channel_mult array
    if (try cm_obj = unet_obj.get("channel_mult")) {
        usz len = cm_obj.get_len();
        for (usz i = 0; i < len && i < 3; i++) {
            unet.channel_mult[i] = (uint)(cm_obj.get_float_at(i) ?? (double)(i + 1));
        }
    }

    // VAE config
    Object* vae_obj = root.get("vae")!;
    VaeConfig vae = {
        .latent_channels = (uint)(vae_obj.get_float("latent_channels") ?? 4.0),
        .out_channels = (uint)(vae_obj.get_float("out_channels") ?? 3.0),
        .prefix_decoder = vae_obj.get_string("prefix_decoder") ?? "first_stage_model.decoder.",
        .prefix_encoder = vae_obj.get_string("prefix_encoder") ?? "first_stage_model.encoder.",
    };

    // Image config
    Object* img_obj = root.get("image")!;
    ImageConfig image = {
        .size = (uint)(img_obj.get_float("size") ?? 512.0),
        .latent_size = (uint)(img_obj.get_float("latent_size") ?? 64.0),
    };

    // Scheduler config
    Object* sched_obj = root.get("scheduler")!;
    SchedulerConfig scheduler = {
        .num_train_timesteps = (uint)(sched_obj.get_float("num_train_timesteps") ?? 1000.0),
        .beta_start = (float)(sched_obj.get_float("beta_start") ?? 0.00085),
        .beta_end = (float)(sched_obj.get_float("beta_end") ?? 0.012),
    };

    io::printfn("\n=== Diffusion Config ===");
    io::printfn("  CLIP: dim=%d, layers=%d, heads=%d, vocab=%d", clip.dim, clip.n_layers, clip.n_heads, clip.vocab_size);
    io::printfn("  UNet: channels=%d, mult=[%d,%d,%d], res_blocks=%d", unet.model_channels,
        unet.channel_mult[0], unet.channel_mult[1], unet.channel_mult[2], unet.num_res_blocks);
    io::printfn("  VAE: latent=%d, out=%d", vae.latent_channels, vae.out_channels);
    io::printfn("  Image: %dx%d (latent %dx%d)", image.size, image.size, image.latent_size, image.latent_size);

    return {
        .name = name,
        .clip = clip,
        .unet = unet,
        .vae = vae,
        .image = image,
        .scheduler = scheduler,
    };
}

fn DiffusionKernels? create_diffusion_kernels(llm::DeviceContext* ctx) {
    io::printfn("Creating diffusion compute kernels...");
    char[] spv = &DIFFUSION_SPV;
    vk::ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    DiffusionKernels kernels = {
        .conv2d           = llm::create_kernel(ctx, shader, 4, Conv2dPC.sizeof, "conv2d")!!,
        .conv2d_q8        = llm::create_kernel(ctx, shader, 4, Conv2dPC.sizeof, "conv2d_q8")!!,
        .group_norm       = llm::create_kernel(ctx, shader, 4, GroupNormPC.sizeof, "group_norm")!!,
        .batched_matmul   = llm::create_kernel(ctx, shader, 3, BatchedMatMulPC.sizeof, "batched_matmul")!!,
        .batched_matmul_q8 = llm::create_kernel(ctx, shader, 3, BatchedMatMulPC.sizeof, "batched_matmul_q8")!!,
        .spatial_attention = llm::create_kernel(ctx, shader, 5, SpatialAttentionPC.sizeof, "spatial_attention")!!,
        .cross_attention  = llm::create_kernel(ctx, shader, 5, CrossAttentionPC.sizeof, "cross_attention")!!,
        .upsample_nearest = llm::create_kernel(ctx, shader, 2, UpsamplePC.sizeof, "upsample_nearest")!!,
        .timestep_embed   = llm::create_kernel(ctx, shader, 1, TimestepEmbedPC.sizeof, "timestep_embed")!!,
        .broadcast_add    = llm::create_kernel(ctx, shader, 2, BroadcastAddPC.sizeof, "broadcast_add")!!,
        .channel_concat   = llm::create_kernel(ctx, shader, 3, ChannelConcatPC.sizeof, "channel_concat")!!,
        .ddim_step        = llm::create_kernel(ctx, shader, 2, DdimStepPC.sizeof, "ddim_step")!!,
        .euler_step       = llm::create_kernel(ctx, shader, 2, EulerStepPC.sizeof, "euler_step")!!,
        .scale_shift_clamp = llm::create_kernel(ctx, shader, 2, ScaleShiftPC.sizeof, "scale_shift_clamp")!!,
        .f16_to_f32       = llm::create_kernel(ctx, shader, 2, F16ToF32PC.sizeof, "f16_to_f32")!!,
        .relu             = llm::create_kernel(ctx, shader, 1, ReluPC.sizeof, "relu")!!,
        .tanh_clamp       = llm::create_kernel(ctx, shader, 1, TanhClampPC.sizeof, "tanh_clamp")!!,
        .shared           = llm::create_shared_kernels(ctx, shader)!!,
    };

    shader.free(ctx.device);
    return kernels;
}

fn void DiffusionKernels.free(&self, vk::Device device) {
    self.conv2d.free(device);
    self.conv2d_q8.free(device);
    self.group_norm.free(device);
    self.batched_matmul.free(device);
    self.batched_matmul_q8.free(device);
    self.spatial_attention.free(device);
    self.cross_attention.free(device);
    self.upsample_nearest.free(device);
    self.timestep_embed.free(device);
    self.broadcast_add.free(device);
    self.channel_concat.free(device);
    self.ddim_step.free(device);
    self.euler_step.free(device);
    self.scale_shift_clamp.free(device);
    self.f16_to_f32.free(device);
    self.relu.free(device);
    self.tanh_clamp.free(device);
    self.shared.free(device);
}

fn void dispatch_conv2d(
    vk::CommandBuffer cmd,
    DiffusionKernels* k,
    llm::Tensor* weight,
    llm::Tensor* bias,
    llm::Tensor* input,
    llm::Tensor* output,
    Conv2dPC* pc
) {
    llm::ComputeKernel* kernel;
    if (weight.dtype == llm::GGML_Q8_0) {
        kernel = &k.conv2d_q8;
    } else {
        kernel = &k.conv2d;
    }
    uint spatial_groups = llm::ceil_div(pc.out_h * pc.out_w, 256);
    llm::dispatch_kernel(cmd, kernel,
        { weight.gpu_buffer.buffer, bias.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { weight.size_bytes, bias.size_bytes, input.size_bytes, output.size_bytes },
        pc, spatial_groups, pc.out_c);
}

fn void dispatch_batched_matmul_auto(
    vk::CommandBuffer cmd,
    DiffusionKernels* k,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    BatchedMatMulPC* pc
) {
    llm::ComputeKernel* kernel;
    if (weight.dtype == llm::GGML_Q8_0) {
        kernel = &k.batched_matmul_q8;
    } else {
        kernel = &k.batched_matmul;
    }
    llm::dispatch_kernel(cmd, kernel,
        { input.gpu_buffer.buffer, weight.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { input.size_bytes, weight.size_bytes, output.size_bytes },
        pc, llm::ceil_div(pc.m_dim, 16), llm::ceil_div(pc.n_dim, 16));
}

