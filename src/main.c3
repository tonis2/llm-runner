module llm;

import std::io;
import vk;
import image;
import image::png;
import llm::args;
import llm::pipelines;
import llm::diffusers;

fn int main(String[] args) {
    // Parse arguments
    AppConfig config = args::parse_args(args)!!;

    io::printfn("Loading model from: %s", config.model_path);
    io::printfn("Prompt: %s", config.prompt);

    // Setup Vulkan context
    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    // Load pipeline (auto-detects model type from GGUF)
    llm::pipelines::ImagePipeline pipeline = llm::pipelines::pipeline_from_gguf(&ctx, config.model_path)!!;
    defer pipeline.free();

    // Configure pipeline
    if (config.text_model_path.len > 0) {
        pipeline.set_text_encoder(config.text_model_path);
    }
    if (config.vae_path.len > 0) {
        pipeline.set_vae(config.vae_path);
    }
    if (config.taesd_path.len > 0) {
        pipeline.set_taesd(config.taesd_path);
    }

    // Prepare inputs
    llm::pipelines::GenerationInputs inputs = llm::pipelines::generation_inputs_defaults();
    inputs.prompt = config.prompt;
    inputs.image_size = config.image_size;
    inputs.num_steps = config.num_steps;
    inputs.cfg_scale = config.cfg_scale;
    inputs.seed = config.seed;

    // Optional: img2img mode
    image::Image input_img;
    if (config.input_path.len > 0) {
        input_img = load_png(config.input_path)!!;
        inputs.input_image = &input_img;
        inputs.strength = 0.75;
    }

    // Generate image
    io::printfn("\nGenerating image...");
    image::Image output = pipeline.generate(&inputs)!!;

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
