module example;

import std::io;
import vk;
import image;
import image::png;
import llm;
import llm::pipelines;
import zimage;

fn int main(String[] args) {
    // Parse arguments
    AppConfig config = parse_args(args)!!;

    io::printfn("Loading model from: %s", config.model_path);
    io::printfn("Prompt: %s", config.prompt);

    // Setup Vulkan context
    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    // Configure pipeline options
    llm::pipelines::PipelineOptions opts = {
        .text_model_path = config.text_model_path,
        .vae_path = config.vae_path,
        .taesd_path = config.taesd_path,
        .offload = config.offload,
    };

    // Create and load pipeline
    llm::pipelines::Pipeline pipeline = zimage::create_pipeline();
    pipeline.load(&ctx, config.model_path, &opts)!!;
    defer pipeline.free();

    // Prepare inputs
    GenerationInputs inputs = pipelines::generation_inputs_defaults();
    inputs.prompt = config.prompt;
    inputs.image_size = config.image_size;
    inputs.num_steps = config.num_steps;
    inputs.cfg_scale = config.cfg_scale;
    inputs.seed = config.seed;

    // Optional: img2img mode
    image::Image input_img;
    if (config.input_path.len > 0) {
        input_img = llm::load_png(config.input_path)!!;
        inputs.input_image = &input_img;
        inputs.strength = 0.75;
    }

    // Generate image
    io::printfn("\nGenerating image...");
    image::Image output = pipeline.generate(&ctx, &inputs)!!;

    // Cleanup input if loaded
    if (config.input_path.len > 0) {
        input_img.free();
    }

    // Save output
    if (catch err = png::save_file(&output, config.output_path)) {
        io::printfn("Error: Failed to save image to %s", config.output_path);
        output.free();
        return 1;
    }

    io::printfn("Image saved to: %s", config.output_path);

    // Cleanup
    output.free();

    return 0;
}
