module llm::pipelines;

import vk;
import std::io;
import std::core::mem;
import llm;
import image;

// --- Embedded Shaders ---

const char[*] DIFFUSION_SPV = $embed("../shaders/diffusion.spv");

// --- Fault Definitions ---

faultdef PIPELINE_DEVICE_ERROR,
         PIPELINE_GENERATION_FAILED,
         PIPELINE_INVALID_INPUT;

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

// --- Shared Push Constant Structs ---
// Used by multiple pipeline implementations (SDXS, ZImage VAE)

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

struct UpsamplePC {
    uint channels;
    uint in_h;
    uint in_w;
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

struct ReluPC {
    uint n;
}

struct TanhClampPC {
    uint n;
}

// --- Scheduler Config ---
// Kept in engine because init_scheduler() references it.

struct SchedulerConfig {
    uint num_train_timesteps;
    float beta_start;
    float beta_end;
}



// ============================================================
// Shared Helpers (used by diffuser implementations)
// ============================================================

// Convert float[0,1] tensor to image::Image (NCHW layout)
fn image::Image? float_tensor_to_image(float[] data, uint width, uint height, uint channels) {
    PixelFormat format;
    switch (channels) {
        case 1: format = PixelFormat.GRAYSCALE;
        case 3: format = PixelFormat.RGB;
        case 4: format = PixelFormat.RGBA;
        default: return PIPELINE_INVALID_INPUT~;
    }

    char[] pixels = mem::new_array(char, (usz)width * height * channels);

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint pixel_idx = y * width + x;
            for (uint c = 0; c < channels; c++) {
                float val = data[c * width * height + pixel_idx];
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                pixels[(usz)pixel_idx * channels + c] = (char)(val * 255.0f + 0.5f);
            }
        }
    }

    return {
        .width = width,
        .height = height,
        .bit_depth = 8,
        .format = format,
        .pixels = pixels,
    };
}

// Convert float tensor to image with SD-style normalization
fn image::Image? float_tensor_to_image_sd(float[] data, uint width, uint height, uint channels) {
    PixelFormat format;
    switch (channels) {
        case 1: format = PixelFormat.GRAYSCALE;
        case 3: format = PixelFormat.RGB;
        case 4: format = PixelFormat.RGBA;
        default: return PIPELINE_INVALID_INPUT~;
    }

    char[] pixels = mem::new_array(char, (usz)width * height * channels);

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint pixel_idx = y * width + x;
            for (uint c = 0; c < channels; c++) {
                float val = data[c * width * height + pixel_idx];
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                pixels[(usz)pixel_idx * channels + c] = (char)(val * 255.0f + 0.5f);
            }
        }
    }

    return {
        .width = width,
        .height = height,
        .bit_depth = 8,
        .format = format,
        .pixels = pixels,
    };
}