<*
 The 54 numbers per point that SkinTokens' mesh encoder takes as input.

 Its `input_proj` is `[512, 54]`, and the 54 decompose exactly as Michelangelo's
 `FourierEmbedder` lays them out for `num_freqs = 8` plus three point features:

     [ x y z | sin(x*f), sin(y*f), sin(z*f) | cos(x*f), cos(y*f), cos(z*f) | nx ny nz ]
        3            8          8        8         8          8        8       3

 with `f = 1, 2, 4, ... 128` — powers of two, no pi factor, since the checkpoint
 records `mesh_encoder_include_pi = False`. Order is not cosmetic: these are
 columns of a trained weight matrix, and permuting them silently produces a
 plausible-looking encoding of nothing.

 Raw xyz is kept alongside the sinusoids so the network still has an unwrapped
 coordinate to work from; the sinusoids only supply the high-frequency detail
 that a linear layer on three inputs cannot represent.
*>
module llm::mesh;

import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::vector;

// From the checkpoint's hyper-parameters: mesh_encoder_num_freqs,
// mesh_encoder_point_feats, mesh_encoder_num_latents, sampler_num_samples.
const uint DEFAULT_NUM_FREQS = 8;
const usz POINT_FEATS = 3;
const usz DEFAULT_LATENTS = 256;
const usz DEFAULT_SAMPLES = 54000;

<*
 Width of the Fourier part alone: xyz plus a sin and a cos per axis per
 frequency. 51 at eight frequencies.
*>
fn usz fourier_dim(uint num_freqs) => 3 * (2 * (usz)num_freqs + 1);

<*
 Full input width of the mesh encoder — Fourier plus the point features. 54 at
 eight frequencies with normals, which is what `input_proj` expects.
*>
fn usz encoder_input_dim(uint num_freqs = DEFAULT_NUM_FREQS, usz point_feats = POINT_FEATS)
    => fourier_dim(num_freqs) + point_feats;

<*
 Write one point's encoder input into `dst`.

 @param dst : `At least encoder_input_dim() long; only that prefix is written.`
 @param include_pi : `Scale the frequencies by pi. False for this checkpoint.`
 @return? BUFFER_TOO_SMALL : `dst cannot hold a full row.`
*>
fn void? embed_point(float[] dst, Vec3 position, Vec3 normal, uint num_freqs = DEFAULT_NUM_FREQS, bool include_pi = false) {
    usz freqs = (usz)num_freqs;
    usz dim = encoder_input_dim(num_freqs, POINT_FEATS);
    if (dst.len < dim) return BUFFER_TOO_SMALL~;

    dst[0] = position[0];
    dst[1] = position[1];
    dst[2] = position[2];

    // sin block first, then the whole cos block: torch concatenates the two
    // full tensors rather than interleaving them per frequency.
    usz sin_base = 3;
    usz cos_base = 3 + 3 * freqs;

    for (usz axis = 0; axis < 3; axis++) {
        float coordinate = position[axis];
        float frequency = include_pi ? (float)math::PI : 1.0f;

        for (usz f = 0; f < freqs; f++) {
            float theta = coordinate * frequency;
            dst[sin_base + axis * freqs + f] = math::sin(theta);
            dst[cos_base + axis * freqs + f] = math::cos(theta);
            frequency *= 2.0f;
        }
    }

    usz feats_base = 3 + 6 * freqs;
    dst[feats_base] = normal[0];
    dst[feats_base + 1] = normal[1];
    dst[feats_base + 2] = normal[2];
}

<*
 Embed a whole cloud into one row-major `count x encoder_input_dim()` array.

 @return? NORMAL_COUNT_MISMATCH : `Positions and normals are different lengths.`
 @return? EMPTY_MESH : `The cloud is empty.`
*>
fn float[]? embed_points(Allocator allocator, PointCloud cloud, uint num_freqs = DEFAULT_NUM_FREQS, bool include_pi = false) {
    if (cloud.positions.len == 0) return EMPTY_MESH~;
    if (cloud.positions.len != cloud.normals.len) return NORMAL_COUNT_MISMATCH~;

    usz dim = encoder_input_dim(num_freqs, POINT_FEATS);
    float[] features = alloc::new_array(allocator, float, (sz)(cloud.positions.len * dim));
    defer catch alloc::free(allocator, features);

    for (usz i = 0; i < cloud.positions.len; i++) {
        embed_point(features[i * dim : dim], cloud.positions[i], cloud.normals[i], num_freqs, include_pi)!;
    }

    return features;
}

enum FitMode {
    UNIT_CUBE,
    UNIT_SPHERE,
}

struct PrepareOptions {
    usz samples;
    usz latents;
    uint num_freqs;
    bool include_pi;
    ulong seed;
    FitMode fit;
    float target;
}

<*
 What the checkpoint's own hyper-parameters imply.

 `samples` is `sampler_num_samples`, the cloud size the model was trained
 against. `fit` is a guess: the checkpoint records no normalisation, and only
 the training repo settles it. Everything else is read straight off the weights.
*>
const PrepareOptions DEFAULT_PREPARE = {
    .samples = DEFAULT_SAMPLES,
    .latents = DEFAULT_LATENTS,
    .num_freqs = DEFAULT_NUM_FREQS,
    .include_pi = false,
    .seed = 0,
    .fit = UNIT_CUBE,
    .target = 1.0f,
};

<*
 Everything the mesh encoder consumes, plus the transform needed to read its
 answers back.

 `features` are the cross-attention keys and values; `query` is the gather of
 `features` at `query_indices`, which is what the encoder's single cross
 attention block attends *from*. Both are row-major, `dim` floats per row.
*>
struct ShapeEncoderInput {
    float[] features;
    float[] query;
    usz[] query_indices;
    usz count;
    usz latents;
    usz dim;
    Normalization normalization;
}

fn void ShapeEncoderInput.free(&self, Allocator allocator) {
    if (self.features.len > 0) alloc::free(allocator, self.features);
    if (self.query.len > 0) alloc::free(allocator, self.query);
    if (self.query_indices.len > 0) alloc::free(allocator, self.query_indices);
    *self = {};
}

<*
 Mesh in, encoder input out: normalise, sample, embed, and pick the queries.

 `geometry` is taken by value and not modified — the normalisation is applied to
 the sampled points instead, and returned in the result so that vertex positions
 handed to the skin decoder later can be put in the same space, and predicted
 bone positions taken back out of it.

 @return? EMPTY_MESH : `The mesh has no triangles.`
 @return? NOT_TRIANGULATED : `The index count is not a multiple of three.`
 @return? NORMAL_COUNT_MISMATCH : `Some but not all vertices carry a normal.`
 @return? INDEX_OUT_OF_RANGE : `An index names a vertex that does not exist.`
 @return? NO_SURFACE_AREA : `Every triangle is degenerate.`
 @return? TOO_FEW_POINTS : `Fewer samples requested than latents needed.`
*>
fn ShapeEncoderInput? prepare_shape_input(Allocator allocator, TriMesh geometry, PrepareOptions options = DEFAULT_PREPARE) {
    if (options.latents > options.samples) return TOO_FEW_POINTS~;

    // Derived from the mesh, applied to the cloud. Taking it from the mesh
    // makes it independent of the sample count and the seed; applying it after
    // sampling is equivalent to normalising first, because a uniform scale
    // commutes with the sampler's barycentric interpolation and leaves the unit
    // normals alone.
    Normalization transform = options.fit == UNIT_SPHERE
        ? geometry.unit_sphere_transform(options.target)
        : geometry.unit_cube_transform(options.target);

    PointCloud cloud = sample_surface(allocator, geometry, options.samples, options.seed)!;
    defer cloud.free(allocator);

    foreach (&position : cloud.positions) {
        *position = transform.forward(*position);
    }

    usz dim = encoder_input_dim(options.num_freqs, POINT_FEATS);

    ShapeEncoderInput input = {
        .count = cloud.positions.len,
        .latents = options.latents,
        .dim = dim,
        .normalization = transform,
    };
    defer catch input.free(allocator);

    input.features = embed_points(allocator, cloud, options.num_freqs, options.include_pi)!;
    input.query_indices = farthest_point_sample(allocator, cloud.positions, options.latents)!;
    input.query = alloc::new_array(allocator, float, (sz)(options.latents * dim));

    foreach (usz slot, index : input.query_indices) {
        for (usz c = 0; c < dim; c++) {
            input.query[slot * dim + c] = input.features[index * dim + c];
        }
    }

    return input;
}
