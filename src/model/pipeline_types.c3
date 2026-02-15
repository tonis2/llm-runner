module llm::pipeline;

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

// Progress callback type - called during generation
// step: current step (1-based), total_steps: total number of steps
// Returns: true to continue, false to abort
// Usage: fn bool my_callback(int step, int total_steps) { ... }
// Then: inputs.on_progress = &my_callback;

// Input struct for Z-Image pipeline (unified txt2img/img2img)
struct ZImageInputs {
    String prompt;           // Required: text prompt
    uint image_size;         // Default: 512
    uint num_steps;          // Default: 4
    float cfg_scale;         // Default: 7.0
    uint seed;               // Default: 42
    
    // Optional: for img2img mode (if null, does txt2img)
    image::Image* input_image;
    float strength;          // 0.0-1.0, only used with input_image
    
    // Optional: external model paths
    String vae_path;         // Path to VAE safetensors
    String taesd_path;       // Path to TAESD safetensors
    String text_model_path;  // Path to text encoder GGUF
}

// Input struct for Stable Diffusion pipeline
struct DiffusionInputs {
    String prompt;
    uint num_steps;          // Default: 20
    float cfg_scale;         // Default: 7.5
    uint seed;               // Default: 42
    
    // Optional: for img2img
    image::Image* input_image;
    float strength;          // Default: 0.75
}

// Default input values
fn ZImageInputs zimage_inputs_defaults() {
    return {
        .prompt = "",
        .image_size = 512,
        .num_steps = 4,
        .cfg_scale = 7.0,
        .seed = 42,
        .input_image = null,
        .strength = 0.75,
        .vae_path = "",
        .taesd_path = "",
        .text_model_path = "",
    };
}

fn DiffusionInputs diffusion_inputs_defaults() {
    return {
        .prompt = "",
        .num_steps = 20,
        .cfg_scale = 7.5,
        .seed = 42,
        .input_image = null,
        .strength = 0.75,
    };
}

// Validate inputs
fn bool ZImageInputs.validate(ZImageInputs* self) {
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

fn bool DiffusionInputs.validate(DiffusionInputs* self) {
    if (self.prompt.len == 0) {
        io::printfn("Error: prompt is required");
        return false;
    }
    if (self.num_steps == 0) {
        io::printfn("Error: num_steps must be > 0");
        return false;
    }
    return true;
}
