module llm::pipelines;

import vk;
import std::io;
import std::io::file;
import std::core::mem;
import image;
import image::png;
import llm;

// Fault definitions for pipeline errors
faultdef PIPELINE_LOAD_FAILED,
         PIPELINE_DEVICE_ERROR,
         PIPELINE_GENERATION_FAILED,
         PIPELINE_INVALID_INPUT,
         PIPELINE_NOT_INITIALIZED;

// --- Pluggable Pipeline Interface ---
// Diffusers implement this interface. The engine delegates to the active impl.

alias CanHandleFn = fn bool(llm::GGUFFile* gf);
alias LoadFn = fn void?(void* data, llm::DeviceContext* ctx, String model_path, PipelineOptions* opts);
alias GenerateFn = fn image::Image?(void* data, llm::DeviceContext* ctx, GenerationInputs* inputs);
alias FreeFn = fn void(void* data);

struct PipelineImpl {
    void* data;                // Opaque pointer to diffuser state
    CanHandleFn can_handle;    // Can this diffuser handle this model file?
    LoadFn load;               // Load model weights to GPU
    GenerateFn generate;       // Generate image from inputs
    FreeFn free_fn;            // Free all GPU resources
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

// --- Pipeline Registry ---
// Engine maintains a list of registered pipeline implementations.
// detect_pipeline() tries each until one returns can_handle=true.

const MAX_PIPELINES = 8;

struct PipelineRegistry {
    PipelineImpl[MAX_PIPELINES] impls;
    uint count;
}

fn void PipelineRegistry.register(PipelineRegistry* self, PipelineImpl impl) {
    if (self.count < MAX_PIPELINES) {
        self.impls[self.count] = impl;
        self.count++;
    }
}

fn PipelineImpl? PipelineRegistry.detect(PipelineRegistry* self, llm::GGUFFile* gf) {
    for (uint i = 0; i < self.count; i++) {
        if (self.impls[i].can_handle(gf)) {
            return self.impls[i];
        }
    }
    return PIPELINE_LOAD_FAILED~;
}

// Global registry
PipelineRegistry g_registry;

fn void register_pipeline(PipelineImpl impl) {
    g_registry.register(impl);
}

fn PipelineImpl? detect_pipeline(String model_path) {
    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();
    llm::GGUFFile gf = llm::gguf_parse(data)!!;
    defer gf.free();
    defer mm.destroy();

    return g_registry.detect(&gf);
}
