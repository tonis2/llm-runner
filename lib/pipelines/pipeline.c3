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

// ImagePipeline: Thin wrapper around Pipeline interface
// Usage:
//   Pipeline pipeline = zimage::create_pipeline();
//   ImagePipeline image_pipeline = create_image_pipeline(&ctx, pipeline, "model.gguf", &opts)!!;
//   image::Image output = image_pipeline.generate(&inputs)!!;
//   image_pipeline.free();

struct ImagePipeline {
    llm::DeviceContext* ctx;
    Pipeline pipeline;
    bool loaded;
}

fn ImagePipeline? create_image_pipeline(llm::DeviceContext* ctx, Pipeline pipeline, String model_path, PipelineOptions* opts) {
    pipeline.load(ctx, model_path, opts)!!;

    return {
        .ctx = ctx,
        .pipeline = pipeline,
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
    return self.pipeline.generate(self.ctx, inputs);
}

fn void ImagePipeline.free(ImagePipeline* self) {
    if (self.loaded) {
        self.pipeline.free();
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
