module llm;

import skintokens;
import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::core::test;

// Scratch probe: beam search at the widths that used to die.
//
// Beams >= 6 crashed in `vk::MemoryAllocator.free` after producing correct
// output, and only on some inputs. That was the temp-arena page list; see
// test/allocator_test.c3. This walks 2, 4, 6, 8 and 10 beams on a real mesh to
// confirm the symptom is gone rather than merely rarer.
//
//     c3c test --trust=full --test-filter test_zz_beam_widths

const String BEAM_GGUF = "test/models/skintokens/grpo_1400.gguf";
const String BEAM_MESH = "/Users/tonis/Downloads/cow_3d.glb";

fn void test_zz_beam_widths() @test
{
    if (!file::is_file(BEAM_GGUF) || !file::is_file(BEAM_MESH))
    {
        io::printfn("  skipped: needs %s and %s", BEAM_GGUF, BEAM_MESH);
        return;
    }

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    mmap::FileMmap weights = file::mmap_open(BEAM_GGUF, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    String schema = (String)&llm_text::QWEN3_CONFIG_JSON;
    ModelConfig config = load_arch_config(schema, &gf)!!;
    defer config.free();
    WeightNames names = load_weight_names(schema)!!;
    defer names.free();

    skintokens::MeshEncoder encoder = skintokens::load_mesh_encoder(&ctx, BEAM_GGUF)!!;
    defer encoder.free();
    llm_text::LlmModel model = llm_text::load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    skintokens::TriMesh geometry = skintokens::load_gltf(mem, BEAM_MESH)!!;
    defer geometry.free(mem);

    skintokens::ShapeEncoderInput shape =
        skintokens::prepare_shape_input(mem, geometry, skintokens::DEFAULT_PREPARE)!!;
    defer shape.free(mem);

    float[] prefix = encoder.encode(mem, &shape)!!;
    defer mem::free(prefix);
    encoder.release_activations();

    usz[5] widths = { 2, 4, 6, 8, 10 };
    foreach (beams : widths)
    {
        skintokens::BeamOptions options = skintokens::DEFAULT_BEAM;
        options.beams = beams;
        options.rig_class = ARTICULATION;

        if (try rig = skintokens::generate_rig_beam(mem, &ctx, &model, prefix, shape.normalization, options))
        {
            defer rig.free(mem);
            io::printfn("  %d beams: %d tokens -> %d bones", beams, rig.tokens.len, rig.skeleton.len());
        }
        else
        {
            io::printfn("  %d beams: no beam finished", beams);
        }
        (void)io::stdout().flush();
    }
}
