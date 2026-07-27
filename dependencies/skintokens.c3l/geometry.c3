<*
 Triangle meshes, and the rigid normalisation a shape encoder wants in front of
 them.

 SkinTokens' mesh encoder is a Michelangelo/3DShape2VecSet perceiver: it does
 not see the mesh at all, only a cloud of points sampled off its surface. So the
 job here is narrow — hold a triangle soup, put it in a canonical box, and hand
 it to `sample_surface`.

 Normalisation is returned rather than just applied, because the model answers
 in the space it was asked in. Bone positions come back normalised, and mapping
 them onto the original mesh is `Normalization.inverse`.
*>
module skintokens;

import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::vector;

faultdef
    EMPTY_MESH,
    NOT_TRIANGULATED,
    NORMAL_COUNT_MISMATCH,
    INDEX_OUT_OF_RANGE,
    NO_SURFACE_AREA,
    TOO_FEW_POINTS,
    BUFFER_TOO_SMALL;

<*
 A triangle soup: three indices per triangle, all pointing into `positions`.

 `normals` is either empty or exactly as long as `positions`. When it is empty
 sampling falls back to face normals, which is correct but faceted — the encoder
 sees a flat-shaded version of the surface.
*>
struct TriMesh {
    Vec3[] positions;
    Vec3[] normals;
    uint[] indices;
}

fn usz TriMesh.triangle_count(self) => self.indices.len / 3;

fn bool TriMesh.has_normals(self) => self.normals.len > 0 && self.normals.len == self.positions.len;

fn void TriMesh.free(&self, Allocator allocator) {
    if (self.positions.len > 0) alloc::free(allocator, self.positions);
    if (self.normals.len > 0) alloc::free(allocator, self.normals);
    if (self.indices.len > 0) alloc::free(allocator, self.indices);
    *self = {};
}

<*
 Check the invariants everything downstream assumes.

 Called by `sample_surface`, so it is only worth calling directly when a mesh is
 built by hand and the error should surface at construction time.

 @return? EMPTY_MESH : `No vertices or no triangles.`
 @return? NOT_TRIANGULATED : `The index count is not a multiple of three.`
 @return? NORMAL_COUNT_MISMATCH : `Some but not all vertices carry a normal.`
 @return? INDEX_OUT_OF_RANGE : `An index names a vertex that does not exist.`
*>
fn void? TriMesh.validate(self) {
    if (self.positions.len == 0 || self.indices.len == 0) return EMPTY_MESH~;
    if (self.indices.len % 3 != 0) return NOT_TRIANGULATED~;
    if (self.normals.len != 0 && self.normals.len != self.positions.len) return NORMAL_COUNT_MISMATCH~;

    foreach (index : self.indices) {
        if ((usz)index >= self.positions.len) return INDEX_OUT_OF_RANGE~;
    }
}

struct Bounds {
    Vec3 min;
    Vec3 max;
}

fn Vec3 Bounds.center(self) => (self.min + self.max) * 0.5f;
fn Vec3 Bounds.extent(self) => self.max - self.min;

fn Bounds TriMesh.bounds(self) {
    if (self.positions.len == 0) return {};

    Bounds box = { .min = self.positions[0], .max = self.positions[0] };
    foreach (position : self.positions) {
        for (usz axis = 0; axis < 3; axis++) {
            if (position[axis] < box.min[axis]) box.min[axis] = position[axis];
            if (position[axis] > box.max[axis]) box.max[axis] = position[axis];
        }
    }
    return box;
}

<*
 The centre-and-uniform-scale that was applied before encoding.

 Uniform, not per-axis: the encoder is trained on shapes whose proportions were
 preserved, and a per-axis fit would stretch a limb into a shape the model has
 never seen. It also keeps normals valid without a separate inverse-transpose.
*>
struct Normalization {
    Vec3 center;
    float scale;
}

fn Vec3 Normalization.forward(self, Vec3 point) => (point - self.center) * self.scale;
fn Vec3 Normalization.inverse(self, Vec3 point) => point * (1.0f / self.scale) + self.center;

fn void TriMesh.apply(&self, Normalization transform) {
    foreach (&position : self.positions) {
        *position = transform.forward(*position);
    }
}

<*
 Centre on the bounding box and scale so the longest axis spans `2 * target`.

 Computed, not applied — the sampled cloud is usually what gets transformed,
 while the mesh stays in its own coordinates. A mesh with no extent at all gets
 scale 1 rather than an infinity.
*>
fn Normalization TriMesh.unit_cube_transform(self, float target = 1.0f) {
    Bounds box = self.bounds();
    Vec3 extent = box.extent();

    float half = math::max(math::max(extent[0], extent[1]), extent[2]) * 0.5f;
    return { .center = box.center(), .scale = half > 0 ? target / half : 1.0f };
}

<*
 Centre on the bounding box and scale so the farthest vertex sits at `target`.

 The sphere fit is the tighter of the two for anything limbed — a character in a
 T-pose wastes most of its bounding box — but which one SkinTokens trained with
 is not recorded in the checkpoint, so both are offered rather than guessed at.
*>
fn Normalization TriMesh.unit_sphere_transform(self, float target = 1.0f) {
    Bounds box = self.bounds();
    Vec3 center = box.center();

    float radius = 0;
    foreach (position : self.positions) {
        float distance = (position - center).length();
        if (distance > radius) radius = distance;
    }

    return { .center = center, .scale = radius > 0 ? target / radius : 1.0f };
}

fn Normalization TriMesh.fit_unit_cube(&self, float target = 1.0f) {
    Normalization transform = self.unit_cube_transform(target);
    self.apply(transform);
    return transform;
}

fn Normalization TriMesh.fit_unit_sphere(&self, float target = 1.0f) {
    Normalization transform = self.unit_sphere_transform(target);
    self.apply(transform);
    return transform;
}

<*
 Total area of every triangle, in whatever units the positions are in.
*>
fn double TriMesh.surface_area(self) {
    double total = 0;
    usz triangles = self.triangle_count();

    for (usz t = 0; t < triangles; t++) {
        Vec3 a = self.positions[self.indices[t * 3]];
        Vec3 b = self.positions[self.indices[t * 3 + 1]];
        Vec3 c = self.positions[self.indices[t * 3 + 2]];
        total += (double)(b - a).cross(c - a).length() * 0.5;
    }
    return total;
}
