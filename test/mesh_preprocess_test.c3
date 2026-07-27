module llm;

import skintokens;
import std::io;
import std::io::file;
import std::core::mem;
import std::core::test;
import std::math;
import std::math::vector;

// A unit cube centred on the origin: 12 triangles, total area 6, every face
// axis-aligned so a sampled normal can be checked exactly.
Vec3[8] cube_positions = {
    { -0.5, -0.5, -0.5 }, { 0.5, -0.5, -0.5 }, { 0.5, 0.5, -0.5 }, { -0.5, 0.5, -0.5 },
    { -0.5, -0.5,  0.5 }, { 0.5, -0.5,  0.5 }, { 0.5, 0.5,  0.5 }, { -0.5, 0.5,  0.5 },
};

uint[36] cube_indices = {
    0, 2, 1,  0, 3, 2,   // -z
    4, 5, 6,  4, 6, 7,   // +z
    0, 4, 7,  0, 7, 3,   // -x
    1, 2, 6,  1, 6, 5,   // +x
    0, 1, 5,  0, 5, 4,   // -y
    3, 7, 6,  3, 6, 2,   // +y
};

fn skintokens::TriMesh unit_cube()
{
    return { .positions = cube_positions[..], .indices = cube_indices[..] };
}

fn bool approx(float a, float b, float epsilon = 1e-5f) => math::abs(a - b) <= epsilon;

fn void test_fourier_embed_layout() @test
{
    // 3 xyz + 3 axes * 8 sin + 3 axes * 8 cos + 3 normal.
    assert(skintokens::fourier_dim(8) == 51);
    assert(skintokens::encoder_input_dim() == 54, "input_proj is [512, 54]");

    Vec3 position = { 0.25, -0.5, 0.75 };
    Vec3 normal = { 0, 1, 0 };

    float[54] row;
    skintokens::embed_point(row[..], position, normal)!!;

    assert(approx(row[0], 0.25f) && approx(row[1], -0.5f) && approx(row[2], 0.75f),
        "raw coordinates lead");

    // Frequencies are powers of two with no pi factor, laid out axis-major:
    // all eight of x, then all eight of y, then z -- and the whole sin block
    // before the whole cos block.
    for (usz axis = 0; axis < 3; axis++)
    {
        float frequency = 1.0f;
        for (usz f = 0; f < 8; f++)
        {
            float theta = position[axis] * frequency;
            assert(approx(row[3 + axis * 8 + f], math::sin(theta)), "sin block");
            assert(approx(row[27 + axis * 8 + f], math::cos(theta)), "cos block");
            frequency *= 2.0f;
        }
    }

    assert(approx(row[51], 0.0f) && approx(row[52], 1.0f) && approx(row[53], 0.0f),
        "point features trail");
}

fn void test_embed_point_rejects_short_buffer() @test
{
    float[53] row;
    assert(@catch(skintokens::embed_point(row[..], { 0, 0, 0 }, { 0, 0, 1 })) == skintokens::BUFFER_TOO_SMALL);
}

fn void test_surface_area() @test
{
    skintokens::TriMesh cube = unit_cube();
    assert(cube.triangle_count() == 12);
    assert(math::abs(cube.surface_area() - 6.0) < 1e-5);
}

fn void test_samples_lie_on_the_surface() @test
{
    skintokens::PointCloud cloud = skintokens::sample_surface(mem, unit_cube(), 4000)!!;
    defer cloud.free(mem);

    assert(cloud.len() == 4000);

    foreach (i, position : cloud.positions)
    {
        float extreme = math::max(math::max(math::abs(position[0]), math::abs(position[1])), math::abs(position[2]));
        assert(approx(extreme, 0.5f), "every point sits on a face of the cube");

        Vec3 normal = cloud.normals[i];
        assert(approx(normal.length(), 1.0f), "normals are unit length");

        // No vertex normals were supplied, so these are face normals: one
        // component is +-1 and the other two vanish.
        float longest = math::max(math::max(math::abs(normal[0]), math::abs(normal[1])), math::abs(normal[2]));
        assert(approx(longest, 1.0f), "face normals are axis aligned");
    }
}

fn void test_normals_default_to_the_face() @test
{
    // A single triangle in the z = 0 plane, given vertex normals that all lean
    // away from it. The face normal is +z; any interpolation of the vertex
    // normals is not.
    Vec3[3] positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
    Vec3[3] leaning = { { 1, 0, 0 }, { 0, 1, 0 }, { 0.6, 0.8, 0 } };
    uint[3] indices = { 0, 1, 2 };

    skintokens::TriMesh sloped = {
        .positions = positions[..],
        .normals = leaning[..],
        .indices = indices[..],
    };
    assert(sloped.has_normals());

    // SkinTokens pairs every sample with the normal of the face it landed on,
    // so that is the default here even when vertex normals are available.
    skintokens::PointCloud flat = skintokens::sample_surface(mem, sloped, 64)!!;
    defer flat.free(mem);
    foreach (normal : flat.normals)
    {
        assert(approx(normal[2], 1.0f), "the face normal is +z");
    }

    skintokens::PointCloud smooth = skintokens::sample_surface(mem, sloped, 64, 0, true)!!;
    defer smooth.free(mem);
    foreach (normal : smooth.normals)
    {
        assert(approx(normal[2], 0.0f), "interpolated normals stay in the plane");
        assert(approx(normal.length(), 1.0f), "and are renormalised");
    }
}

fn void test_sampling_is_area_weighted() @test
{
    // Two disjoint squares in the z = 0 plane, one four times the area of the
    // other, separated so a sample can be attributed by the sign of x.
    Vec3[8] positions = {
        { -3, 0, 0 }, { -1, 0, 0 }, { -1, 2, 0 }, { -3, 2, 0 },   // area 4
        {  1, 0, 0 }, {  2, 0, 0 }, {  2, 1, 0 }, {  1, 1, 0 },   // area 1
    };
    uint[12] indices = { 0, 1, 2,  0, 2, 3,  4, 5, 6,  4, 6, 7 };

    skintokens::TriMesh quads = { .positions = positions[..], .indices = indices[..] };
    assert(math::abs(quads.surface_area() - 5.0) < 1e-5);

    skintokens::PointCloud cloud = skintokens::sample_surface(mem, quads, 20000)!!;
    defer cloud.free(mem);

    usz big = 0;
    foreach (position : cloud.positions)
    {
        if (position[0] < 0) big++;
    }

    // 4/5 of the area, so 16000 expected. The seed is fixed, so this is a
    // regression check on the weighting rather than a flaky statistical one.
    assert(big > 15600 && big < 16400, "large square takes its share of samples");
}

fn void test_sampling_is_deterministic() @test
{
    skintokens::PointCloud a = skintokens::sample_surface(mem, unit_cube(), 256, 7)!!;
    defer a.free(mem);
    skintokens::PointCloud b = skintokens::sample_surface(mem, unit_cube(), 256, 7)!!;
    defer b.free(mem);
    skintokens::PointCloud c = skintokens::sample_surface(mem, unit_cube(), 256, 8)!!;
    defer c.free(mem);

    usz matches = 0;
    foreach (i, position : a.positions)
    {
        assert(position == b.positions[i], "the same seed gives the same cloud");
        if (position == c.positions[i]) matches++;
    }
    assert(matches < 8, "a different seed gives a different cloud");
}

fn void test_sampling_rejects_bad_meshes() @test
{
    skintokens::TriMesh empty;
    assert(@catch(skintokens::sample_surface(mem, empty, 16)) == skintokens::EMPTY_MESH);

    Vec3[3] positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
    uint[4] ragged = { 0, 1, 2, 0 };
    skintokens::TriMesh not_triangulated = { .positions = positions[..], .indices = ragged[..] };
    assert(@catch(skintokens::sample_surface(mem, not_triangulated, 16)) == skintokens::NOT_TRIANGULATED);

    uint[3] past_the_end = { 0, 1, 3 };
    skintokens::TriMesh bad_index = { .positions = positions[..], .indices = past_the_end[..] };
    assert(@catch(skintokens::sample_surface(mem, bad_index, 16)) == skintokens::INDEX_OUT_OF_RANGE);

    uint[3] degenerate_indices = { 0, 0, 0 };
    skintokens::TriMesh degenerate = { .positions = positions[..], .indices = degenerate_indices[..] };
    assert(@catch(skintokens::sample_surface(mem, degenerate, 16)) == skintokens::NO_SURFACE_AREA);
}

fn void test_farthest_point_sample_takes_the_extremes() @test
{
    Vec3[10] line;
    for (usz i = 0; i < 10; i++) line[i] = { (float)i, 0, 0 };

    usz[] picked = skintokens::farthest_point_sample(mem, line[..], 3)!!;
    defer mem::free(picked);

    assert(picked[0] == 0, "the start is the first pick");
    assert(picked[1] == 9, "the far end is next");
    assert(picked[2] == 4 || picked[2] == 5, "then the middle");
}

fn void test_farthest_point_sample_spreads_over_a_grid() @test
{
    // 5x5 grid over the unit square, index 0 at a corner.
    Vec3[25] grid;
    for (usz y = 0; y < 5; y++)
    {
        for (usz x = 0; x < 5; x++)
        {
            grid[y * 5 + x] = { (float)x * 0.25f, (float)y * 0.25f, 0 };
        }
    }

    usz[] picked = skintokens::farthest_point_sample(mem, grid[..], 4)!!;
    defer mem::free(picked);

    bool[25] seen;
    foreach (index : picked) seen[index] = true;

    assert(seen[0] && seen[4] && seen[20] && seen[24], "all four corners get picked");
}

fn void test_farthest_point_sample_rejects_bad_counts() @test
{
    Vec3[4] points;
    Vec3[] nothing;
    assert(@catch(skintokens::farthest_point_sample(mem, nothing, 1)) == skintokens::EMPTY_MESH);
    assert(@catch(skintokens::farthest_point_sample(mem, points[..], 5)) == skintokens::TOO_FEW_POINTS);
    assert(@catch(skintokens::farthest_point_sample(mem, points[..], 0)) == skintokens::TOO_FEW_POINTS);
    assert(@catch(skintokens::farthest_point_sample(mem, points[..], 2, 4)) == skintokens::INDEX_OUT_OF_RANGE);
}

fn void test_normalization_round_trips() @test
{
    Vec3[3] positions = { { -4, 0, -1 }, { 2, 1, 1 }, { 0, 0.5, 0 } };
    uint[3] indices = { 0, 1, 2 };
    skintokens::TriMesh geometry = { .positions = positions[..], .indices = indices[..] };

    // Extent (6, 1, 2), so the longest half-axis is 3 and the fit scales by 1/3
    // about the box centre (-1, 0.5, 0).
    skintokens::Normalization transform = geometry.unit_cube_transform();
    assert(approx(transform.scale, 1.0f / 3.0f));
    assert(approx(transform.center[0], -1.0f) && approx(transform.center[1], 0.5f));

    Vec3 mapped = transform.forward({ 2, 1, 1 });
    assert(approx(mapped[0], 1.0f), "the widest axis reaches the box edge");

    foreach (position : positions)
    {
        Vec3 back = transform.inverse(transform.forward(position));
        for (usz axis = 0; axis < 3; axis++) assert(approx(back[axis], position[axis]));
    }
}

fn void test_prepare_shape_input() @test
{
    skintokens::PrepareOptions options = skintokens::DEFAULT_PREPARE;
    options.samples = 2000;
    options.latents = 32;   // draws a pool of 128 and keeps 32

    skintokens::TriMesh cube = unit_cube();
    skintokens::ShapeEncoderInput input = skintokens::prepare_shape_input(mem, cube, options)!!;
    defer input.free(mem);

    assert(input.dim == 54);
    assert(input.count == 2000 && input.latents == 32);
    assert(input.features.len == 2000 * 54);
    assert(input.query.len == 32 * 54);
    assert(input.query_indices.len == 32);

    // The query is a gather out of the features, not a second embedding.
    foreach (usz slot, index : input.query_indices)
    {
        assert(index < input.count);
        for (usz c = 0; c < input.dim; c++)
        {
            assert(input.query[slot * input.dim + c] == input.features[index * input.dim + c]);
        }
    }

    // A unit cube fits the normalisation box exactly, so the leading xyz of
    // every row lands in [-1, 1] with the extremes reached.
    float extreme = 0;
    for (usz i = 0; i < input.count; i++)
    {
        for (usz axis = 0; axis < 3; axis++)
        {
            float coordinate = input.features[i * input.dim + axis];
            assert(coordinate >= -1.0001f && coordinate <= 1.0001f, "normalised into the unit box");
            if (math::abs(coordinate) > extreme) extreme = math::abs(coordinate);
        }
    }
    assert(approx(extreme, 1.0f, 1e-3f), "and it fills that box");
}

fn void test_prepare_rejects_a_pool_bigger_than_the_cloud() @test
{
    skintokens::PrepareOptions options = skintokens::DEFAULT_PREPARE;
    options.samples = 16;
    options.latents = 32;
    assert(@catch(skintokens::prepare_shape_input(mem, unit_cube(), options)) == skintokens::TOO_FEW_POINTS);

    // The pool is four times the latents, so this is short even though there
    // are more samples than queries.
    options.samples = 100;
    options.latents = 32;
    assert(@catch(skintokens::prepare_shape_input(mem, unit_cube(), options)) == skintokens::TOO_FEW_POINTS);
}

fn void test_pick_queries_spreads_over_a_uniform_draw() @test
{
    // A line of 400 points. Whatever pool is drawn, farthest point sampling
    // over it still reaches for the ends first.
    Vec3[400] line;
    for (usz i = 0; i < 400; i++) line[i] = { (float)i, 0, 0 };

    usz[] picked = skintokens::pick_queries(mem, line[..], 8, 4, 1)!!;
    defer mem::free(picked);

    assert(picked.len == 8);

    // Distinct cloud indices: the pool is drawn without replacement.
    for (usz a = 0; a < picked.len; a++)
    {
        for (usz b = a + 1; b < picked.len; b++) assert(picked[a] != picked[b], "queries repeat");
    }

    float lowest = 1.0e30f;
    float highest = -1.0e30f;
    foreach (index : picked)
    {
        assert(index < 400);
        if (line[index][0] < lowest) lowest = line[index][0];
        if (line[index][0] > highest) highest = line[index][0];
    }
    // 32 points drawn from 400, so the extremes of the pool land near the ends.
    assert(lowest < 40 && highest > 360, "the picks span the line");

    assert(@catch(skintokens::pick_queries(mem, line[..], 200, 4)) == skintokens::TOO_FEW_POINTS);
}

// The glTF adapter is exercised against a fixture that ships with the gltf
// submodule. Skipped rather than failed if the submodule is not checked out.
const String TRUCK_GLB = "dependencies/gltf.c3l/test/truck.glb";

fn void test_load_gltf() @test
{
    if (try File probe = file::open(TRUCK_GLB, "rb"))
    {
        (void)probe.close();
    }
    else
    {
        io::printfn("  skipped: %s not present", TRUCK_GLB);
        return;
    }

    skintokens::TriMesh geometry = skintokens::load_gltf(mem, TRUCK_GLB)!!;
    defer geometry.free(mem);

    assert(geometry.positions.len > 0);
    assert(geometry.triangle_count() > 0);
    assert(geometry.has_normals(), "normals are read or derived for every vertex");
    assert(geometry.surface_area() > 0);
    geometry.validate()!!;

    foreach (normal : geometry.normals)
    {
        assert(approx(normal.length(), 1.0f, 1e-3f), "normals come out unit length");
    }

    skintokens::PrepareOptions options = skintokens::DEFAULT_PREPARE;
    options.samples = 4096;
    options.latents = 256;

    skintokens::ShapeEncoderInput input = skintokens::prepare_shape_input(mem, geometry, options)!!;
    defer input.free(mem);

    assert(input.features.len == 4096 * 54);
    assert(input.query.len == 256 * 54);

    // Predicted bone positions come back in the normalised frame; the transform
    // is what puts them back on the model.
    Vec3 corner = geometry.bounds().max;
    Vec3 back = input.normalization.inverse(input.normalization.forward(corner));
    for (usz axis = 0; axis < 3; axis++) assert(approx(back[axis], corner[axis], 1e-3f));
}
