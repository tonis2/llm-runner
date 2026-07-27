<*
 Flattening a glTF scene into a single triangle soup.

 The encoder has no notion of parts, materials or hierarchy — it wants one
 surface. So every triangle primitive reachable from the scene is baked through
 its node's world matrix and concatenated, and everything else about the file is
 dropped.

     `TriMesh geometry = mesh::load_gltf(mem, "character.glb")!;`
     `defer geometry.free(mem);`

 Only TRIANGLES primitives are taken. Strips and fans are rare in exported
 assets and are skipped rather than silently mis-triangulated.
*>
module skintokens;

import gltf;
import std::core::mem;
import std::core::mem::allocator;
import std::collections::list;
import std::math;
import std::math::matrix;
import std::math::vector;

faultdef NO_TRIANGLES, UNSUPPORTED_INDEX_TYPE, SCENE_NOT_FOUND;

alias Vec3List = List{Vec3};
alias UintList = List{uint};

// glTF spells the primitive topology as the raw enum value in the attribute
// map. 4 is TRIANGLES, and an absent entry means TRIANGLES too.
const sz GLTF_MODE_TRIANGLES @local = 4;

<*
 Open a .gltf or .glb, flatten its default scene, and close it again.

 @param path : `File to read; .bin sidecars resolve relative to it.`
 @return? NO_TRIANGLES : `Nothing in the scene is a triangle mesh.`
*>
fn TriMesh? load_gltf(Allocator allocator, String path) {
    gltf::GltfStream stream = gltf::open(path, allocator)!;
    defer stream.close();

    stream.load_all_buffers()!;
    return from_gltf(allocator, &stream);
}

<*
 Flatten an already-open stream. Its buffers must be loaded.

 @param scene : `Scene to walk, or -1 for the first one.`
 @return? NO_TRIANGLES : `Nothing in the scene is a triangle mesh.`
 @return? SCENE_NOT_FOUND : `scene names a scene the file does not have.`
*>
fn TriMesh? from_gltf(Allocator allocator, gltf::GltfStream* stream, int scene = -1) {
    if (stream.gltf.scenes.len() == 0) return NO_TRIANGLES~;

    sz scene_index = scene < 0 ? 0 : (sz)scene;
    if (scene_index >= stream.gltf.scenes.len()) return SCENE_NOT_FOUND~;

    Vec3List positions;
    Vec3List normals;
    UintList indices;
    positions.init(allocator);
    normals.init(allocator);
    indices.init(allocator);

    // Freed on every path: on error, and on success too, since the lists are
    // copied into tightly sized arrays before returning.
    defer {
        positions.free();
        normals.free();
        indices.free();
    }

    foreach (root : stream.gltf.scenes[scene_index].nodes) {
        append_node(stream, root, &positions, &normals, &indices)!;
    }

    if (indices.len() == 0) return NO_TRIANGLES~;

    TriMesh geometry = {
        .positions = positions.to_array(allocator),
        .normals = normals.to_array(allocator),
        .indices = indices.to_array(allocator),
    };
    defer catch geometry.free(allocator);

    geometry.validate()!;
    return geometry;
}

// Walks children through global_matrix() rather than threading a transform
// down, because that method already accumulates the whole parent chain.
fn void? append_node(gltf::GltfStream* stream, uint node_index, Vec3List* positions, Vec3List* normals, UintList* indices) @local {
    gltf::Node node = stream.gltf.nodes[node_index];

    if (node.has_mesh()) {
        Matrix4f world = node.global_matrix();

        // Normals need the inverse transpose to survive a non-uniform scale.
        // A singular node matrix has no meaningful normal transform, so the
        // world matrix stands in and the renormalisation below salvages it.
        Matrix4f normal_matrix = world;
        if (try Matrix4f inverted = world.inverse()) normal_matrix = inverted.transpose();

        gltf::Mesh mesh = stream.gltf.meshes[(usz)node.mesh];
        foreach (primitive : mesh.primitives) {
            append_primitive(stream, primitive, world, normal_matrix, positions, normals, indices)!;
        }
    }

    foreach (child : node.children) {
        append_node(stream, child, positions, normals, indices)!;
    }
}

fn void? append_primitive(
    gltf::GltfStream* stream,
    gltf::Primitive primitive,
    Matrix4f world,
    Matrix4f normal_matrix,
    Vec3List* positions,
    Vec3List* normals,
    UintList* indices) @local
{
    if (try sz mode = primitive.attributes.get("mode")) {
        if (mode != GLTF_MODE_TRIANGLES) return;
    }

    sz? position_attribute = primitive.attributes.get("POSITION");
    if (catch position_attribute) return;

    gltf::Accessor position_accessor = stream.gltf.accessors[position_attribute];
    usz vertex_count = (usz)position_accessor.count;
    if (vertex_count == 0) return;

    // Where this primitive's vertices land, so its indices can be rebased and
    // its own normals recomputed without touching earlier primitives.
    uint base = (uint)positions.len();

    for (usz i = 0; i < vertex_count; i++) {
        Vec3 local = stream.@cast_buffer(position_accessor, (sz)i, Vec3);
        positions.push((world * (Vec4){ local[0], local[1], local[2], 1 }).xyz);
    }

    bool supplied_normals = false;
    if (try sz normal_attribute = primitive.attributes.get("NORMAL")) {
        supplied_normals = true;
        gltf::Accessor normal_accessor = stream.gltf.accessors[normal_attribute];
        for (usz i = 0; i < vertex_count; i++) {
            Vec3 local = i < (usz)normal_accessor.count
                ? stream.@cast_buffer(normal_accessor, (sz)i, Vec3)
                : (Vec3){ 0, 0, 0 };
            Vec3 world_normal = (normal_matrix * (Vec4){ local[0], local[1], local[2], 0 }).xyz;
            normals.push(safe_normalize(world_normal));
        }
    } else {
        // Filled in from this primitive's own faces once the indices are known.
        for (usz i = 0; i < vertex_count; i++) normals.push((Vec3){ 0, 0, 0 });
    }

    sz triangle_start = indices.len();

    if (try sz index_attribute = primitive.attributes.get("indices")) {
        gltf::Accessor index_accessor = stream.gltf.accessors[index_attribute];
        usz count = (usz)index_accessor.count - (usz)index_accessor.count % 3;

        for (usz i = 0; i < count; i++) {
            uint index = read_index(stream, index_accessor, (sz)i)!;
            if ((usz)index >= vertex_count) return INDEX_OUT_OF_RANGE~;
            indices.push(base + index);
        }
    } else {
        // Unindexed: the vertex buffer is already in triangle order.
        usz count = vertex_count - vertex_count % 3;
        for (usz i = 0; i < count; i++) indices.push(base + (uint)i);
    }

    if (!supplied_normals) {
        compute_normals(positions, normals, indices, base, vertex_count, triangle_start);
    }
}

// Smooth normals from the primitive's own triangles. The face normals are left
// unnormalised as they accumulate, which weights each face by twice its area --
// the standard trick for stopping a fan of slivers from dominating a corner.
fn void compute_normals(
    Vec3List* positions,
    Vec3List* normals,
    UintList* indices,
    uint base,
    usz vertex_count,
    sz triangle_start) @local
{
    for (sz i = triangle_start; i + 2 < indices.len(); i += 3) {
        sz ia = (sz)indices.get(i);
        sz ib = (sz)indices.get(i + 1);
        sz ic = (sz)indices.get(i + 2);

        Vec3 a = positions.get(ia);
        Vec3 face = (positions.get(ib) - a).cross(positions.get(ic) - a);
        normals.set(ia, normals.get(ia) + face);
        normals.set(ib, normals.get(ib) + face);
        normals.set(ic, normals.get(ic) + face);
    }

    for (sz i = 0; i < (sz)vertex_count; i++) {
        sz at = (sz)base + i;
        normals.set(at, safe_normalize(normals.get(at)));
    }
}

fn Vec3 safe_normalize(Vec3 v) @local {
    float length = v.length();
    if (length < 1e-20f) return { 0, 0, 1 };
    return v * (1.0f / length);
}

fn uint? read_index(gltf::GltfStream* stream, gltf::Accessor accessor, sz i) @local {
    switch (accessor.data_type) {
        case gltf::DataType.CHAR: return (uint)stream.@cast_buffer(accessor, i, char);
        case gltf::DataType.USHORT: return (uint)stream.@cast_buffer(accessor, i, ushort);
        case gltf::DataType.UINT: return stream.@cast_buffer(accessor, i, uint);
        default: return UNSUPPORTED_INDEX_TYPE~;
    }
}
