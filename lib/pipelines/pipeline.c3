module llm::pipelines;

import vk;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;
import std::time::clock;
import image;
import image::png;
import llm;

// ImagePipeline: Thin wrapper around PipelineImpl
// Usage:
//   register_pipeline(zimage::create_pipeline());
//   register_pipeline(sdxs::create_pipeline());
//
//   ImagePipeline pipeline = create_image_pipeline(&ctx, "model.gguf", &opts)!!;
//   image::Image output = pipeline.generate(&inputs)!!;
//   pipeline.free();

struct ImagePipeline {
    llm::DeviceContext* ctx;
    PipelineImpl impl;
    bool loaded;
}

fn ImagePipeline? create_image_pipeline(llm::DeviceContext* ctx, String model_path, PipelineOptions* opts) {
    // Auto-detect pipeline from model file
    PipelineImpl impl = detect_pipeline(model_path)!!;

    // Load the model
    impl.load(impl.data, ctx, model_path, opts)!!;

    return {
        .ctx = ctx,
        .impl = impl,
        .loaded = true,
    };
}

fn ImagePipeline? create_image_pipeline_with_impl(llm::DeviceContext* ctx, PipelineImpl impl, String model_path, PipelineOptions* opts) {
    impl.load(impl.data, ctx, model_path, opts)!!;

    return {
        .ctx = ctx,
        .impl = impl,
        .loaded = true,
    };
}

fn image::Image? ImagePipeline.generate(ImagePipeline* self, GenerationInputs* inputs) {
    if (!self.loaded) {
        io::printfn("Error: Pipeline not loaded");
        return PIPELINE_NOT_INITIALIZED~;
    }
    if (!inputs.validate()) {
        return PIPELINE_INVALID_INPUT~;
    }
    return self.impl.generate(self.impl.data, self.ctx, inputs);
}

fn void ImagePipeline.free(ImagePipeline* self) {
    if (self.loaded) {
        self.impl.free_fn(self.impl.data);
        self.loaded = false;
    }
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
