module depth_main;

import llm;
import depth_anything;
import std::io;

fn int main(String[] args) {
    if (args.len < 3) {
        io::printfn("Usage: depth <depth_anything_v2_vits_fp32.safetensors> <input.jpg> [output.png] [opts]");
        io::printfn("  Monocular depth estimation (Depth Anything V2, ViT-S/14).");
        io::printfn("  Default: grayscale scene-depth map (near = bright). Default output: depth.png");
        io::printfn("  Options:");
        io::printfn("    --res N      inference resolution, longest side in px (default 518).");
        io::printfn("                 Higher = sharper depth but more VRAM (1036 ~720MB, 1554 ~3.6GB).");
        io::printfn("    --height     output a tiling height/relief map for game textures instead of");
        io::printfn("                 scene depth (high-pass detrend + albedo detail). Defaults res to 1036.");
        io::printfn("    --albedo W   use albedo luminance as the height base, 0..1 (default 0 = model depth).");
        io::printfn("                 Use ~1.0 for flat-but-detailed surfaces (wood grain, carving, cracks)");
        io::printfn("                 the depth model can't resolve; 0 for real 3D shape (rounded stones).");
        io::printfn("    --flatten F  how much to equalize albedo tone, 0..1 (default 0.70). 1 = full local");
        io::printfn("                 relief, lower keeps more plank-to-plank tone.");
        io::printfn("    --detail W   extra fine luminance high-pass on top (default 0.30, 0 = off).");
        io::printfn("    --detrend F  high-pass radius as a fraction of the longest side (default 0.10).");
        io::printfn("                 Lower flattens more large-scale shape (~0.06 for very flat tiling).");
        return 1;
    }

    String model_path = args[1];
    String img_path = args[2];
    String out_path = "depth.png";
    bool out_set = false;
    uint res = 0;            // 0 => pick a default below (depends on mode)
    bool height = false;
    float detail = 0.30f;
    float detrend = 0.10f;   // tiling textures want the large-scale scene depth flattened
    float albedo = 0.0f;     // 0 = model depth base; 1 = albedo luminance base
    float flatten = 0.70f;   // how much to equalize albedo tone

    for (usz i = 3; i < args.len; i++) {
        String a = args[i];
        if (a == "--res")          { res = args[i + 1].to_uint() ?? 518; i++; }
        else if (a == "--height")  { height = true; }
        else if (a == "--albedo")  { albedo = args[i + 1].to_float() ?? 1.0f; i++; }
        else if (a == "--flatten") { flatten = args[i + 1].to_float() ?? 0.70f; i++; }
        else if (a == "--detail")  { detail = args[i + 1].to_float() ?? 0.30f; i++; }
        else if (a == "--detrend") { detrend = args[i + 1].to_float() ?? 0.10f; i++; }
        else if (!out_set)         { out_path = a; out_set = true; }
    }
    // Texture height maps benefit from finer detail; default them to 2x resolution.
    if (res == 0) res = height ? 1036 : 518;

    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    depth_anything::DepthModel model = depth_anything::load_depth_model(&ctx, model_path, res)!!;

    depth_anything::PreprocInfo pp = depth_anything::preprocess_image(&ctx, img_path, &model)!!;
    io::printfn("=== Depth estimation: %s (%dx%d) -> grid %dx%d, model input %dx%d ===",
        img_path, pp.orig_w, pp.orig_h, pp.gw, pp.gh, pp.gw * model.config.patch_size, pp.gh * model.config.patch_size);

    model.encode(&pp.image, &pp.pos_embed, pp.gh, pp.gw)!!;
    model.decode(pp.gh, pp.gw)!!;

    uint sh = pp.gh * model.config.patch_size;
    uint sw = pp.gw * model.config.patch_size;
    if (height) {
        depth_anything::save_height_map(&ctx, &model.kernels, &model.dpt.depth,
            sh, sw, pp.orig_w, pp.orig_h, img_path, detrend, detail, albedo, flatten, out_path)!!;
    } else {
        depth_anything::save_depth_png(&ctx, &model.kernels, &model.dpt.depth,
            sh, sw, pp.orig_w, pp.orig_h, out_path)!!;
    }
    return 0;
}
