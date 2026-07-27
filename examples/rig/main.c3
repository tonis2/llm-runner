<*
 Mesh in, rigged .glb out.

     rig cow.glb cow_rigged.glb [--model <gguf>] [--seed N] [--class none|rignet|vroid|articulation]

 The skeleton is the model's. The skin weights are not — those come from the
 checkpoint's skin VAE, which is not ported, so `export_rig_glb` falls back to
 a distance bind. See lib/mesh/rig_export.c3.

 Generation samples, so a run that does not terminate within budget is a normal
 outcome rather than a failure; seeds are retried until one lands, which is what
 the reference buys with its ten beams.
*>
module rig_cli;

import llm;
import llm::mesh;
import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::vector;

const String DEFAULT_MODEL = "/Users/tonis/Documents/ai_models/grpo_1400.gguf";
const uint SEED_ATTEMPTS = 8;

fn mesh::RigClass? parse_class(String text)
{
    switch (text)
    {
        case "none": return mesh::RigClass.NONE;
        case "rignet": return mesh::RigClass.RIGNET;
        case "vroid": return mesh::RigClass.VROID;
        case "articulation": return mesh::RigClass.ARTICULATION;
        default: return io::UNSUPPORTED_OPERATION~;
    }
}

fn int main(String[] args)
{
    if (args.len < 3)
    {
        io::printn("usage: rig <input.glb> <output.glb> [--model <gguf>] [--seed N] [--class <name>]");
        return 1;
    }

    String input = args[1];
    String output = args[2];
    String model_path = DEFAULT_MODEL;
    ulong seed = 0;
    mesh::RigClass rig_class = mesh::RigClass.ARTICULATION;

    for (usz i = 3; i < args.len; i++)
    {
        switch (args[i])
        {
            case "--model":
                if (++i >= args.len) { io::printn("--model needs a path"); return 1; }
                model_path = args[i];
            case "--seed":
                if (++i >= args.len) { io::printn("--seed needs a number"); return 1; }
                seed = args[i].to_ulong() ?? 0;
            case "--class":
                if (++i >= args.len) { io::printn("--class needs a name"); return 1; }
                rig_class = parse_class(args[i]) ?? mesh::RigClass.ARTICULATION;
            default:
                io::printfn("unknown option %s", args[i]);
                return 1;
        }
    }

    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    mmap::FileMmap weights = file::mmap_open(model_path, "rb")!!;
    defer (void)weights.destroy();
    llm::GGUFFile gf = llm::gguf_parse(weights.bytes())!!;
    defer gf.free();

    String schema = (String)&llm_text::QWEN3_CONFIG_JSON;
    llm::ModelConfig config = llm::load_arch_config(schema, &gf)!!;
    defer config.free();
    llm::WeightNames names = llm::load_weight_names(schema)!!;
    defer names.free();

    mesh::MeshEncoder encoder = mesh::load_mesh_encoder(&ctx, model_path)!!;
    defer encoder.free();
    llm_text::LlmModel model = llm_text::load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    mesh::TriMesh geometry = mesh::load_gltf(mem, input)!!;
    defer geometry.free(mem);
    io::printfn("%s: %d vertices, %d triangles", input, geometry.positions.len, geometry.triangle_count());

    // The encoder is the slow half and does not depend on the seed, so it runs
    // once and every attempt re-decodes from the same prefix.
    mesh::PrepareOptions prepare = mesh::DEFAULT_PREPARE;
    mesh::ShapeEncoderInput shape = mesh::prepare_shape_input(mem, geometry, prepare)!!;
    defer shape.free(mem);
    float[] prefix = encoder.encode(mem, &shape)!!;
    defer mem::free(prefix);

    mesh::RigOptions options = mesh::DEFAULT_RIG;
    options.prepare = prepare;
    options.rig_class = rig_class;

    // Sample several seeds and keep the cleanest. The model wanders when a
    // sequence runs long -- the reference does the same -- so the first
    // skeleton that parses is not necessarily one worth exporting. A run with
    // every bone on the mesh is taken immediately; otherwise the least bad wins.
    mesh::Bounds box = geometry.bounds();
    mesh::RigResult rig;
    bool rigged = false;
    float best_off = 2.0f;
    ulong best_seed = seed;

    for (uint attempt = 0; attempt < SEED_ATTEMPTS; attempt++)
    {
        options.seed = seed + attempt;
        if (try candidate = mesh::generate_rig_from_prefix(
            mem, &ctx, &model, prefix, shape.normalization, options))
        {
            float off = candidate.off_mesh_fraction(box);
            io::printfn("  seed %d: %d bones, %.0f%% off mesh",
                options.seed, candidate.skeleton.len(), off * 100.0f);

            if (off < best_off)
            {
                if (rigged) rig.free(mem);
                rig = candidate;
                rigged = true;
                best_off = off;
                best_seed = options.seed;
            }
            else
            {
                candidate.free(mem);
            }
            if (best_off == 0.0f) break;
        }
        else
        {
            io::printfn("  seed %d did not produce a complete skeleton", options.seed);
        }
    }

    if (!rigged)
    {
        io::printfn("no seed produced a skeleton in %d attempts", SEED_ATTEMPTS);
        return 1;
    }
    defer rig.free(mem);

    usz forks = 0;
    usz[] children = mem::new_array(usz, (sz)rig.skeleton.len());
    defer mem::free(children);
    foreach (parent : rig.skeleton.parents) if (parent >= 0) children[(usz)parent]++;
    foreach (count : children) if (count > 1) forks++;

    io::printfn("rig: %d tokens -> %d bones, %d branch points, %.0f%% off mesh (seed %d, class %s)",
        rig.tokens.len, rig.skeleton.len(), forks, best_off * 100.0f, best_seed, rig_class);

    mesh::export_rig_glb(mem, output, geometry, &rig)!!;
    io::printfn("wrote %s", output);
    return 0;
}
