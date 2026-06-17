module depth_main;

import llm;
import depth_anything;
import std::io;

fn int main(String[] args) {
    if (args.len < 3) {
        io::printfn("Usage: depth <depth_anything_v2_vits_fp32.safetensors> <input.jpg> [output.png] [options]");
        io::printfn("  Default: writes a grayscale depth map (near = bright). Output: depth.png");
        io::printfn("");
        io::printfn("Options:");
        io::printfn("  --stereo        Write a side-by-side (L|R) stereo pair for VR (Quest) instead.");
        io::printfn("                  Default output: stereo.png");
        io::printfn("  --disp=FRAC     Max disparity as a fraction of width (default 0.03). Stereo only.");
        io::printfn("  --conv=PLANE    Depth plane [0..1] at the screen (default 1.0 = whole scene behind");
        io::printfn("                  the screen; 0.5 = near pops out). Stereo only.");
        return 1;
    }

    String model_path = args[1];
    String img_path = args[2];

    // Parse flags + optional positional output path.
    bool stereo = false;
    String out_path = "";
    float disp_frac = 0.03f;
    float conv = 1.0f;
    for (usz i = 3; i < args.len; i++) {
        String a = args[i];
        if (a == "--stereo") {
            stereo = true;
        } else if (a.len > 7 && a[0:7] == "--disp=") {
            disp_frac = a[7..].to_float()!!;
        } else if (a.len > 7 && a[0:7] == "--conv=") {
            conv = a[7..].to_float()!!;
        } else if (a.len >= 2 && a[0:2] == "--") {
            io::printfn("Unknown option: %s", a);
            return 1;
        } else {
            out_path = a;
        }
    }
    if (out_path == "") out_path = stereo ? "stereo.png" : "depth.png";

    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    depth_anything::DepthModel model = depth_anything::load_depth_model(&ctx, model_path)!!;

    depth_anything::PreprocInfo pp = depth_anything::preprocess_image(&ctx, img_path, &model)!!;
    io::printfn("=== Depth estimation: %s (%dx%d) -> grid %dx%d, model input %dx%d ===",
        img_path, pp.orig_w, pp.orig_h, pp.gw, pp.gh, pp.gw * model.config.patch_size, pp.gh * model.config.patch_size);

    model.encode(&pp.image, &pp.pos_embed, pp.gh, pp.gw)!!;
    model.decode(pp.gh, pp.gw)!!;

    uint sh = pp.gh * model.config.patch_size;
    uint sw = pp.gw * model.config.patch_size;

    if (stereo) {
        depth_anything::save_stereo_sbs(&ctx, &model.kernels, &model.dpt.depth,
            sh, sw, img_path, pp.orig_w, pp.orig_h, out_path, disp_frac, conv)!!;
    } else {
        depth_anything::save_depth_png(&ctx, &model.kernels, &model.dpt.depth,
            sh, sw, pp.orig_w, pp.orig_h, out_path)!!;
    }
    return 0;
}
