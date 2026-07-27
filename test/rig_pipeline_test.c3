module llm;

import llm::mesh;
import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::core::test;
import std::math;
import std::math::vector;

// Mesh in, bones out: the whole path, from a triangle soup through the shape
// encoder and the transformer to a skeleton in the mesh's own coordinates.
const String RIG_GGUF = "test/models/skintokens/grpo_1400.gguf";

Vec3[8] rig_cube_positions = {
    { -0.5, -0.5, -0.5 }, { 0.5, -0.5, -0.5 }, { 0.5, 0.5, -0.5 }, { -0.5, 0.5, -0.5 },
    { -0.5, -0.5,  0.5 }, { 0.5, -0.5,  0.5 }, { 0.5, 0.5,  0.5 }, { -0.5, 0.5,  0.5 },
};

uint[36] rig_cube_indices = {
    0, 2, 1,  0, 3, 2,   4, 5, 6,  4, 6, 7,
    0, 4, 7,  0, 7, 3,   1, 2, 6,  1, 6, 5,
    0, 1, 5,  0, 5, 4,   3, 7, 6,  3, 6, 2,
};

fn void test_mesh_to_bones() @test
{
    if (try File probe = file::open(RIG_GGUF, "rb"))
    {
        (void)probe.close();
    }
    else
    {
        io::printfn("  skipped: %s not present", RIG_GGUF);
        return;
    }

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    mmap::FileMmap weights = file::mmap_open(RIG_GGUF, "rb")!!;
    defer (void)weights.destroy();
    GGUFFile gf = gguf_parse(weights.bytes())!!;
    defer gf.free();

    String schema = (String)&llm_text::QWEN3_CONFIG_JSON;
    ModelConfig config = load_arch_config(schema, &gf)!!;
    defer config.free();
    WeightNames names = load_weight_names(schema)!!;
    defer names.free();

    mesh::MeshEncoder encoder = mesh::load_mesh_encoder(&ctx, RIG_GGUF)!!;
    defer encoder.free();
    llm_text::LlmModel model = llm_text::load_llm_model(&ctx, &gf, &config, &names)!!;
    defer model.free();

    mesh::TriMesh cube = {
        .positions = rig_cube_positions[..],
        .indices = rig_cube_indices[..],
    };

    mesh::RigOptions options = mesh::DEFAULT_RIG;
    options.prepare.samples = 16384;
    options.max_new_tokens = 600;

    // Sampling one sequence does not always reach an end within budget, and a
    // cube is about as far outside a character rigging model's distribution as
    // an input gets. The reference searches ten beams for exactly this reason;
    // retrying a couple of seeds is the poor relation of that, and it is the
    // caller's call to make rather than the library's.
    mesh::RigResult rig;
    bool generated = false;
    for (ulong seed = 0; seed < 3 && !generated; seed++)
    {
        options.seed = seed;
        if (try attempt = mesh::generate_rig(mem, &ctx, &encoder, &model, cube, options))
        {
            rig = attempt;
            generated = true;
        }
        else
        {
            io::printfn("  seed %d did not terminate within %d tokens", seed, options.max_new_tokens);
        }
    }
    assert(generated, "no seed produced a complete sequence");
    defer rig.free(mem);

    // Terminated on its own rather than running out of budget — that is what
    // the grammar buys, and without it decoding runs past any cap.
    assert(rig.tokens.len < 600, "generation used the whole budget");
    assert(rig.tokens[0] == mesh::TOKEN_BOS);
    assert(rig.tokens[rig.tokens.len - 1] == mesh::TOKEN_EOS
        || rig.tokens[rig.tokens.len - 1] == mesh::TOKEN_SEQUENCE_END,
        "the sequence did not end on a terminator");

    // Nothing outside the skeleton vocabulary can appear: the model is trained
    // to follow the skeleton with skin codes, and only the grammar keeps the
    // two apart.
    foreach (token : rig.tokens)
    {
        assert(!mesh::is_skin_token(token), "a skin token leaked into the skeleton");
        assert(token < mesh::RIG_TOTAL_VOCAB);
    }

    assert(rig.skeleton.len() > 0, "no bones");
    assert(rig.prefix_tokens == 512, "512 shape latents conditioned the sequence");

    // A tree: the root has no parent, everything else hangs off something
    // earlier, so walking parents always terminates.
    assert(rig.skeleton.parents[0] == -1, "the first bone should be the root");
    for (usz i = 1; i < rig.skeleton.len(); i++)
    {
        int parent = rig.skeleton.parents[i];
        assert(parent >= 0 && (usz)parent < i,
            "bone %d has parent %d, which is not an earlier bone", i, parent);
    }

    // Bones come out in the normalised frame; mapping them back has to land
    // them on the mesh, which for this cube means inside its own bounds.
    Vec3[] world = rig.joints_in_model_space(mem);
    defer mem::free(world);

    mesh::Bounds box = cube.bounds();
    foreach (i, joint : world)
    {
        for (usz axis = 0; axis < 3; axis++)
        {
            assert(joint[axis] >= box.min[axis] - 0.01f && joint[axis] <= box.max[axis] + 0.01f,
                "bone %d axis %d at %f is outside the mesh", i, axis, joint[axis]);
        }
    }

    io::printfn("  rig: %d tokens -> %d bones", rig.tokens.len, rig.skeleton.len());
}

// The grammar on its own, without a GPU: it has to keep decoding inside the
// skeleton vocabulary and offer a way to stop.
fn void test_grammar_constrains_decoding() @test
{
    mesh::RigGrammar grammar;

    bool[] allowed = mem::new_array(bool, (sz)mesh::RIG_TOTAL_VOCAB);
    defer mem::free(allowed);

    // Only bos may open a sequence.
    grammar.allow_next(allowed);
    assert(allowed[mesh::TOKEN_BOS]);
    assert(!allowed[0] && !allowed[mesh::TOKEN_EOS]);

    grammar.accept(mesh::TOKEN_BOS);
    grammar.allow_next(allowed);
    assert(allowed[mesh::TOKEN_CLS_VROID] && allowed[0], "a class or a coordinate may follow bos");
    assert(!allowed[mesh::TOKEN_EOS], "an empty skeleton is not a legal sequence");

    grammar.accept(mesh::TOKEN_CLS_VROID);

    // Coordinates come in threes, and nothing else may interrupt them.
    grammar.accept(0);
    grammar.allow_next(allowed);
    assert(allowed[5] && !allowed[mesh::TOKEN_BRANCH] && !allowed[mesh::TOKEN_EOS],
        "the second coordinate of a joint is forced");
    grammar.accept(0);
    grammar.allow_next(allowed);
    assert(allowed[5] && !allowed[mesh::TOKEN_EOS], "so is the third");
    grammar.accept(0);

    // A complete joint: now the sequence may branch, start a part, continue, or end.
    grammar.allow_next(allowed);
    assert(allowed[mesh::TOKEN_BRANCH] && allowed[mesh::TOKEN_EOS]
        && allowed[mesh::TOKEN_PART_HAND] && allowed[7]);

    // At no point is a skin token or the model's own terminator reachable.
    assert(!allowed[mesh::SKIN_TOKEN_BASE]);
    assert(!allowed[mesh::TOKEN_SEQUENCE_END]);

    // A branch is followed by six coordinates, so the first three are forced too.
    grammar.accept(mesh::TOKEN_BRANCH);
    grammar.allow_next(allowed);
    assert(allowed[9] && !allowed[mesh::TOKEN_EOS] && !allowed[mesh::TOKEN_BRANCH]);
}
