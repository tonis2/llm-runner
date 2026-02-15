module llm::pipelines;

import vk;
import std::io;
import std::io::file;
import std::core::mem;
import image;
import image::png;

// Fault definitions for pipeline errors
faultdef PIPELINE_LOAD_FAILED,
         PIPELINE_DEVICE_ERROR,
         PIPELINE_GENERATION_FAILED,
         PIPELINE_INVALID_INPUT,
         PIPELINE_NOT_INITIALIZED;

// --- Component Kind Enums ---

enum TextEncoderKind : char {
    NONE,
    CLIP,
    LLM,
}

enum DenoiserKind : char {
    NONE,
    UNET,
    DIT,
}

enum DecoderKind : char {
    NONE,
    SD_VAE,
    FLUX_VAE,
    TAESD,
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
