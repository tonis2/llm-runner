<*
 Turning a mesh into the point cloud a shape encoder reads.

 Two steps, both standard for this family of models:

 - `sample_surface` scatters points over the triangles in proportion to their
   area, so a large flat panel does not end up as sparsely covered as a dense
   cluster of tiny triangles. Tessellation stops mattering; only shape does.
 - `farthest_point_sample` picks a well-spread subset of those points. The mesh
   encoder has no learned query parameters — its 256 latents *are* the embedded
   features at these indices — so this is not an optimisation, it is where the
   queries come from.
*>
module llm::mesh;

import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::random;
import std::math::vector;

<*
 Points sampled off a surface, each with a unit normal.
*>
struct PointCloud {
    Vec3[] positions;
    Vec3[] normals;
}

fn usz PointCloud.len(self) => self.positions.len;

fn void PointCloud.free(&self, Allocator allocator) {
    if (self.positions.len > 0) alloc::free(allocator, self.positions);
    if (self.normals.len > 0) alloc::free(allocator, self.normals);
    *self = {};
}

// First index whose cumulative area is strictly past `target`. Strictly, so
// that a zero-area triangle -- which leaves the running total unchanged -- can
// never be selected: its slot always compares equal to its predecessor's.
fn usz cdf_upper_bound(double[] cdf, double target) @local {
    usz low = 0;
    usz high = cdf.len - 1;

    while (low < high) {
        usz mid = low + (high - low) / 2;
        if (cdf[mid] <= target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

fn Vec3 triangle_normal(Vec3 a, Vec3 b, Vec3 c) @local {
    Vec3 normal = (b - a).cross(c - a);
    float length = normal.length();
    // Only reachable for a degenerate triangle, which area weighting will not
    // pick. Some direction still has to be returned rather than a NaN.
    if (length < 1e-20f) return { 0, 0, 1 };
    return normal * (1.0f / length);
}

<*
 Scatter `count` points over the mesh, area-weighted.

 A triangle is drawn with probability proportional to its area, then a point is
 drawn uniformly inside it by folding a unit square onto the triangle. The
 normal is barycentrically interpolated from the vertex normals when the mesh
 has them, and is the face normal otherwise.

 Sampling is fully determined by `seed`: the same mesh and seed give the same
 cloud, which matters because the encoder's latents are gathered from these
 points and a rig that changes run to run is not debuggable.

 @param count : `Number of points to draw; SkinTokens sampled 54000.`
 @return? EMPTY_MESH : `The mesh has no triangles.`
 @return? NOT_TRIANGULATED : `The index count is not a multiple of three.`
 @return? NORMAL_COUNT_MISMATCH : `Some but not all vertices carry a normal.`
 @return? INDEX_OUT_OF_RANGE : `An index names a vertex that does not exist.`
 @return? NO_SURFACE_AREA : `Every triangle is degenerate.`
 @return? TOO_FEW_POINTS : `count is zero.`
*>
fn PointCloud? sample_surface(Allocator allocator, TriMesh geometry, usz count, ulong seed = 0) {
    geometry.validate()!;
    if (count == 0) return TOO_FEW_POINTS~;

    usz triangles = geometry.triangle_count();

    // Cumulative area in double. A 50k-triangle mesh with a wide spread of
    // triangle sizes accumulates enough rounding in float that the tail of the
    // table stops being monotonic, and a non-monotonic table breaks the search.
    double[] cdf = alloc::new_array(allocator, double, (sz)triangles);
    defer alloc::free(allocator, cdf);

    double total = 0;
    for (usz t = 0; t < triangles; t++) {
        Vec3 a = geometry.positions[geometry.indices[t * 3]];
        Vec3 b = geometry.positions[geometry.indices[t * 3 + 1]];
        Vec3 c = geometry.positions[geometry.indices[t * 3 + 2]];
        total += (double)(b - a).cross(c - a).length() * 0.5;
        cdf[t] = total;
    }
    if (total <= 0) return NO_SURFACE_AREA~;

    PointCloud cloud = {
        .positions = alloc::new_array(allocator, Vec3, (sz)count),
        .normals = alloc::new_array(allocator, Vec3, (sz)count),
    };

    bool interpolate = geometry.has_normals();

    DefaultRandom rng;
    random::seed(&rng, seed);

    for (usz i = 0; i < count; i++) {
        usz t = cdf_upper_bound(cdf, random::next_double(&rng) * total);

        uint ia = geometry.indices[t * 3];
        uint ib = geometry.indices[t * 3 + 1];
        uint ic = geometry.indices[t * 3 + 2];

        Vec3 a = geometry.positions[ia];
        Vec3 b = geometry.positions[ib];
        Vec3 c = geometry.positions[ic];

        // Draw in the unit square and fold the far half back across the
        // diagonal. Uniform over the triangle, and no rejection loop.
        float u = (float)random::next_double(&rng);
        float v = (float)random::next_double(&rng);
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        float w = 1.0f - u - v;

        cloud.positions[i] = a * w + b * u + c * v;

        Vec3 normal = triangle_normal(a, b, c);
        if (interpolate) {
            Vec3 blended = geometry.normals[ia] * w + geometry.normals[ib] * u + geometry.normals[ic] * v;
            // Opposed vertex normals can cancel on a crease. The face normal is
            // the honest answer there.
            if (blended.length() > 1e-6f) normal = blended.normalize();
        }
        cloud.normals[i] = normal;
    }

    return cloud;
}

<*
 Greedily pick `count` well-spread points: each new pick is the one farthest
 from everything picked so far.

 Deterministic, unlike the training-time implementation — torch_cluster's `fps`
 starts from a random index by default. A fixed start costs nothing in coverage
 (the second pick is the farthest point in the cloud regardless) and makes the
 encoder's queries reproducible.

 Returns indices into `points`, in pick order, so a caller can gather the
 matching rows out of an already-embedded feature array.

 @param count : `Number of points to pick; the mesh encoder wants 256.`
 @param start : `Index of the first pick.`
 @return? EMPTY_MESH : `No points to pick from.`
 @return? TOO_FEW_POINTS : `count is zero or larger than the cloud.`
 @return? INDEX_OUT_OF_RANGE : `start is not a valid index.`
*>
fn usz[]? farthest_point_sample(Allocator allocator, Vec3[] points, usz count, usz start = 0) {
    if (points.len == 0) return EMPTY_MESH~;
    if (count == 0 || count > points.len) return TOO_FEW_POINTS~;
    if (start >= points.len) return INDEX_OUT_OF_RANGE~;

    usz[] picked = alloc::new_array(allocator, usz, (sz)count);

    // Squared distance from each point to the nearest pick so far. Squared
    // throughout: the argmax is the same and it saves a sqrt per point per
    // round, which at 54000 x 256 is most of the work.
    float[] nearest = alloc::new_array(allocator, float, (sz)points.len);
    defer alloc::free(allocator, nearest);
    foreach (&distance : nearest) *distance = math::FLOAT_MAX;

    usz current = start;
    for (usz k = 0; k < count; k++) {
        picked[k] = current;

        Vec3 anchor = points[current];
        float farthest = -1.0f;
        usz next = current;

        for (usz i = 0; i < points.len; i++) {
            float distance = (points[i] - anchor).length_sq();
            if (distance < nearest[i]) nearest[i] = distance;
            if (nearest[i] > farthest) {
                farthest = nearest[i];
                next = i;
            }
        }
        current = next;
    }

    return picked;
}
