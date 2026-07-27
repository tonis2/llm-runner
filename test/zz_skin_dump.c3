module llm;

import skintokens;
import std::io;
import std::io::file;
import std::core::mem;
import std::core::test;
import std::math;
import std::math::vector;

// Dump the skin decoder's inputs and its answer, so that
// `tools/compare_skin_reference.py` can run the same thing in torch.
//
// The kernels have CPU references in mesh_encoder_test; what those cannot
// catch is a misread of the architecture — an eps, a norm applied to the wrong
// side of a cross attention, or the fused-projection head split, which is
// wrong in a way that stays finite and plausible. So this feeds a fixed input
// through the real weights and hands both sides the same bytes.
//
//     c3c test --trust=full --test-filter test_zz_dump_skin_reference
//     rigvenv/bin/python tools/compare_skin_reference.py
//
// Named zz_ because it is a fixture generator rather than an assertion: it
// checks that the pipeline runs and writes files, and the comparison happens
// in the tool.

const String SKIN_GGUF = "test/models/skintokens/grpo_1400.gguf";
const String SKIN_DUMP = "build";

// Small enough that the torch side finishes in a minute, large enough that
// the conditioning encoder's farthest point sampling has something to pick
// from: 384 cond tokens want a pool of 1536, and the shape encoder's 512 want
// 2048.
const usz DUMP_SAMPLES = 8192;
const usz DUMP_BONES = 3;

fn bool have_skin_weights()
{
    if (try File f = file::open(SKIN_GGUF, "rb"))
    {
        (void)f.close();
        return true;
    }
    io::printfn("  skipped: %s not present", SKIN_GGUF);
    return false;
}

// A sphere rather than a cube: the decoder is a field over position and
// normal, and a cube gives it six normals to work with.
fn skintokens::TriMesh dump_sphere(usz rings, usz segments)
{
    usz vertices = (rings + 1) * (segments + 1);
    Vec3[] positions = mem::new_array(Vec3, (sz)vertices);
    Vec3[] normals = mem::new_array(Vec3, (sz)vertices);
    uint[] indices = mem::new_array(uint, (sz)(rings * segments * 6));

    for (usz r = 0; r <= rings; r++)
    {
        double phi = math::PI * (double)r / (double)rings;
        for (usz s = 0; s <= segments; s++)
        {
            double theta = 2.0 * math::PI * (double)s / (double)segments;
            Vec3 unit = {
                (float)(math::sin(phi) * math::cos(theta)),
                (float)math::cos(phi),
                (float)(math::sin(phi) * math::sin(theta)),
            };
            usz i = r * (segments + 1) + s;
            // Not a unit sphere: an anisotropic one exercises the
            // normalisation, and a sphere of radius 1 would hide a scale bug.
            positions[i] = { unit.x * 0.7f, unit.y * 1.3f, unit.z * 0.5f };
            normals[i] = unit;
        }
    }

    usz t = 0;
    for (usz r = 0; r < rings; r++)
    {
        for (usz s = 0; s < segments; s++)
        {
            uint a = (uint)(r * (segments + 1) + s);
            uint b = (uint)((r + 1) * (segments + 1) + s);
            indices[t] = a;     indices[t + 1] = b;     indices[t + 2] = a + 1;
            indices[t + 3] = a + 1; indices[t + 4] = b; indices[t + 5] = b + 1;
            t += 6;
        }
    }

    return { .positions = positions, .normals = normals, .indices = indices };
}

fn void write_floats(String name, float[] data)
{
    @pool()
    {
        String path = string::tformat("%s/%s", SKIN_DUMP, name);
        File f = file::open(path, "wb")!!;
        defer (void)f.close();
        f.write(((char*)data.ptr)[:data.len * 4])!!;
    };
}

fn void write_ints(String name, uint[] data)
{
    @pool()
    {
        String path = string::tformat("%s/%s", SKIN_DUMP, name);
        File f = file::open(path, "wb")!!;
        defer (void)f.close();
        f.write(((char*)data.ptr)[:data.len * 4])!!;
    };
}

fn void write_vec3s(String name, Vec3[] data)
{
    float[] flat = mem::new_array(float, (sz)(data.len * 3));
    defer mem::free(flat);
    foreach (usz i, v : data)
    {
        flat[i * 3] = v.x;
        flat[i * 3 + 1] = v.y;
        flat[i * 3 + 2] = v.z;
    }
    write_floats(name, flat);
}

fn void test_zz_dump_skin_reference() @test
{
    if (!have_skin_weights()) return;

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    skintokens::SkinDecoder decoder = skintokens::load_skin_decoder(&ctx, SKIN_GGUF)!!;
    defer decoder.free();
    skintokens::TriMesh sphere = dump_sphere(12, 24);
    defer sphere.free(mem);

    skintokens::PrepareOptions prepare = skintokens::DEFAULT_PREPARE;
    prepare.samples = DUMP_SAMPLES;
    prepare.seed = 7;

    skintokens::ShapeEncoderInput shape = skintokens::prepare_shape_input(mem, sphere, prepare)!!;
    defer shape.free(mem);

    // The same call `encode_condition` makes, with the same arguments. It is
    // deterministic in the cloud and the seed, so reproducing it here gives
    // the torch side the exact points the decoder conditioned on.
    usz[] cond_indices = skintokens::pick_queries(
        mem, shape.cloud.positions, skintokens::VAE_COND_TOKENS, skintokens::QUERY_POOL_RATIO, 0)!!;
    defer mem::free(cond_indices);

    // Stand-ins for what the transformer would emit. Spread over the codebook
    // rather than clustered, so a mistake in the mixed-radix unpacking shows
    // up as a difference rather than as the same wrong latent everywhere.
    uint[] codes = mem::new_array(uint, (sz)(DUMP_BONES * skintokens::TOKENS_PER_SKIN));
    defer mem::free(codes);
    foreach (usz i, &code : codes) *code = (uint)((i * 7919 + 12345) % 32768);

    // Query at the sphere's own vertices, normalised into the frame the
    // encoder saw.
    Vec3[] points = mem::new_array(Vec3, (sz)sphere.positions.len);
    defer mem::free(points);
    foreach (usz i, position : sphere.positions) points[i] = shape.normalization.forward(position);

    float[] skin = decoder.decode(mem, &shape, codes, DUMP_BONES, points, sphere.normals)!!;
    defer mem::free(skin);

    assert(skin.len == sphere.positions.len * DUMP_BONES);

    float lowest = skin[0];
    float highest = skin[0];
    double total = 0;
    foreach (value : skin)
    {
        assert(value >= 0.0f && value <= 1.0f, "sigmoid output out of range");
        if (value < lowest) lowest = value;
        if (value > highest) highest = value;
        total += (double)value;
    }

    uint[] cond_flat = mem::new_array(uint, (sz)cond_indices.len);
    defer mem::free(cond_flat);
    foreach (usz i, index : cond_indices) cond_flat[i] = (uint)index;

    write_vec3s("skin_cloud_pos.f32", shape.cloud.positions);
    write_vec3s("skin_cloud_nrm.f32", shape.cloud.normals);
    write_ints("skin_cond_idx.u32", cond_flat);
    write_ints("skin_codes.u32", codes);
    write_vec3s("skin_query_pos.f32", points);
    write_vec3s("skin_query_nrm.f32", sphere.normals);
    write_floats("skin_mine.f32", skin);

    io::printfn("  dumped %d points, %d bones, %d queries -> %s/skin_mine.f32",
        shape.cloud.positions.len, DUMP_BONES, points.len, SKIN_DUMP);
    io::printfn("  weights min %.5f max %.5f mean %.5f",
        lowest, highest, total / (double)skin.len);
}
