<*
 Mesh in, rigged .glb out.

     rig cow.glb cow_rigged.glb [--model <gguf>] [--seed N] [--class none|rignet|vroid|articulation] [--no-skin]

 Both halves are the model's: the skeleton comes out of the transformer and so
 do the skin weights, as four codes per bone that the VAE decoder turns back
 into a weight per vertex. `--no-skin` skips the second half and falls back to
 the distance bind in `rig_export.c3`.

 Generation samples, so a run that does not terminate within budget is a normal
 outcome rather than a failure; seeds are retried until one lands, which is what
 the reference buys with its ten beams. The search runs skeleton-only — skin
 codes cost more tokens than the skeleton they follow, and eight seeds' worth
 of them would be thrown away — and the winner is then decoded once more with
 skin. Same seed, same sampler, same prefix, so the skeleton comes back
 identical and decoding simply carries on past its terminator.
*>
module rig_cli;

import llm;
import skintokens;
import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::vector;

const String DEFAULT_MODEL = "/Users/tonis/Documents/ai_models/grpo_1400.gguf";
const uint SEED_ATTEMPTS = 8;

fn skintokens::RigClass? parse_class(String text)
{
    switch (text)
    {
        case "none": return skintokens::RigClass.NONE;
        case "rignet": return skintokens::RigClass.RIGNET;
        case "vroid": return skintokens::RigClass.VROID;
        case "articulation": return skintokens::RigClass.ARTICULATION;
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
    bool want_skin = true;
    skintokens::RigClass rig_class = skintokens::RigClass.ARTICULATION;

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
                rig_class = parse_class(args[i]) ?? skintokens::RigClass.ARTICULATION;
            case "--no-skin":
                want_skin = false;
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

    skintokens::MeshEncoder encoder = skintokens::load_mesh_encoder(&ctx, model_path)!!;
    defer encoder.free();
    llm_text::LlmModel model = llm_text::load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    skintokens::TriMesh geometry = skintokens::load_gltf(mem, input)!!;
    defer geometry.free(mem);
    io::printfn("%s: %d vertices, %d triangles", input, geometry.positions.len, geometry.triangle_count());

    // The encoder is the slow half and does not depend on the seed, so it runs
    // once and every attempt re-decodes from the same prefix.
    skintokens::PrepareOptions prepare = skintokens::DEFAULT_PREPARE;
    skintokens::ShapeEncoderInput shape = skintokens::prepare_shape_input(mem, geometry, prepare)!!;
    defer shape.free(mem);
    float[] prefix = encoder.encode(mem, &shape)!!;
    defer mem::free(prefix);

    skintokens::RigOptions options = skintokens::DEFAULT_RIG;
    options.prepare = prepare;
    options.rig_class = rig_class;
    options.predict_skin = false;

    // Sample several seeds and keep the cleanest. The model wanders when a
    // sequence runs long -- the reference does the same -- so the first
    // skeleton that parses is not necessarily one worth exporting. A run with
    // every bone on the mesh is taken immediately; otherwise the least bad wins.
    skintokens::Bounds box = geometry.bounds();
    skintokens::RigResult rig;
    bool rigged = false;
    float best_off = 2.0f;
    ulong best_seed = seed;

    for (uint attempt = 0; attempt < SEED_ATTEMPTS; attempt++)
    {
        options.seed = seed + attempt;
        if (try candidate = skintokens::generate_rig_from_prefix(
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

    // The tokenizer can mark where a part starts -- `spring` for the secondary
    // chains a character rig hangs off itself, `body` and `hand` for the two
    // halves a naming template is keyed on. We parse them and have never looked
    // at them, and they are the model's own word for what a run of joints is.
    foreach (usz i, token : rig.skeleton.part_tokens)
    {
        String part = "part";
        switch (token)
        {
            case skintokens::TOKEN_SPRING: part = "spring";
            case skintokens::TOKEN_PART_BODY: part = "body";
            case skintokens::TOKEN_PART_HAND: part = "hand";
            default: break;
        }
        io::printfn("  part: %s starts at bone %d", part, rig.skeleton.part_starts[i]);
    }

    // The encoder is done with; its activations are the biggest thing on the
    // device and the skin decoder is about to want the space.
    encoder.release_activations();

    float[] skin = {};
    defer if (skin.len > 0) mem::free(skin);

    if (want_skin)
    {
        options.seed = best_seed;
        options.predict_skin = true;

        if (try full = skintokens::generate_rig_from_prefix(
            mem, &ctx, &model, prefix, shape.normalization, options))
        {
            if (full.has_skin() && full.skeleton.len() == rig.skeleton.len())
            {
                skintokens::SkinDecoder decoder = skintokens::load_skin_decoder(&ctx, model_path)!!;
                defer decoder.free();
                skin = decoder.decode_rig(mem, &shape, &full, geometry)!!;

                // The skinned run supersedes the search result — same seed, so
                // the same skeleton, and now the weights that go with it. It
                // has to replace `rig` rather than sit alongside it, because
                // pruning rewrites the skeleton and the weight matrix together
                // and the export needs both halves to agree.
                rig.free(mem);
                rig = full;

                usz dropped = skintokens::prune_unused_joints(
                    mem, &rig.skeleton, &skin, geometry.positions.len)!!;

                io::printfn("skin: %d codes -> %d x %d weights",
                    rig.skin_count, geometry.positions.len, rig.skeleton.len());
                if (dropped > 0)
                {
                    io::printfn("  dropped %d joints too few vertices lean on -> %d bones",
                        dropped, rig.skeleton.len());
                }
            }
            else
            {
                full.free(mem);
                io::printn("skin: the model did not finish a skin block; falling back to a distance bind");
            }
        }
        else
        {
            io::printn("skin: the skin pass did not terminate; falling back to a distance bind");
        }
    }

    skintokens::export_rig_glb(mem, output, geometry, &rig, skin: skin)!!;
    io::printfn("wrote %s", output);
    return 0;
}
