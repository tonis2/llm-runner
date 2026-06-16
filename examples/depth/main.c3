module depth_main;

import llm;
import depth_anything;
import std::io;

fn int main(String[] args) {
    if (args.len < 3) {
        io::printfn("Usage: depth <depth_anything_v2_vits_fp32.safetensors> <input.jpg> [output.png]");
        io::printfn("  Monocular depth estimation (Depth Anything V2, ViT-S/14).");
        io::printfn("  Writes a grayscale depth map (near = bright). Default output: depth.png");
        return 1;
    }

    String model_path = args[1];
    String img_path = args[2];
    String out_path = args.len > 3 ? args[3] : "depth.png";

    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    depth_anything::DepthModel model = depth_anything::load_depth_model(&ctx, model_path)!!;

    depth_anything::PreprocInfo pp = depth_anything::preprocess_image(&ctx, img_path, &model)!!;
    io::printfn("=== Depth estimation: %s (%dx%d) -> grid %dx%d, model input %dx%d ===",
        img_path, pp.orig_w, pp.orig_h, pp.gw, pp.gh, pp.gw * model.config.patch_size, pp.gh * model.config.patch_size);

    model.encode(&pp.image, &pp.pos_embed, pp.gh, pp.gw)!!;
    model.decode(pp.gh, pp.gw)!!;

    depth_anything::save_depth_png(&ctx, &model.kernels, &model.dpt.depth,
        pp.gh * model.config.patch_size, pp.gw * model.config.patch_size, pp.orig_w, pp.orig_h, out_path)!!;
    return 0;
}
