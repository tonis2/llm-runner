<*
 Writing a generated rig out as a .glb.

 `generate_rig` predicts a skeleton and nothing else. glTF skinning needs a
 weight per bone per vertex as well, and those come from the checkpoint's skin
 VAE, which is not ported — so `bind_geometric` stands in for it with a plain
 distance falloff. That is a placeholder, not a prediction: the skeleton in the
 exported file is the model's, the weights are not.

     `RigResult rig = mesh::generate_rig(mem, &ctx, &encoder, &model, geometry)!;`
     `mesh::export_rig_glb(mem, "cow_rigged.glb", geometry, &rig)!;`
*>
module skintokens;

import gltf;
import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::vector;

<*
 Squared distance from a point to the segment between a bone and its parent.

 Segments rather than joints because a limb's vertices sit alongside the bone,
 not around its ends: measured to the nearest joint, a vertex halfway down a
 thigh is equidistant from hip and knee and ends up blended between them, which
 makes the whole limb bend from the middle. The root, having no parent, is a
 point and is measured as one.
*>
fn float segment_distance_sq(Vec3 point, Vec3 head, Vec3 tail) @local
{
    Vec3 axis = tail - head;
    float length_sq = axis.dot(axis);
    if (length_sq <= 1.0e-12f)
    {
        Vec3 d = point - head;
        return d.dot(d);
    }
    float t = (point - head).dot(axis) / length_sq;
    t = math::clamp(t, 0.0f, 1.0f);
    Vec3 d = point - (head + axis * t);
    return d.dot(d);
}

<*
 A dense vertex-by-bone weight matrix from geometry alone.

 Each vertex is weighted towards each bone by `1 / (d^2 + eps)`, so influence
 falls off with the square of the distance to the bone's segment and no vertex
 is ever left with nothing. `sparsify_weights` reduces this to the four
 strongest influences and renormalises, so only the relative magnitudes matter
 and the absolute scale is free.

 The caller owns the returned array. `eps` sets how soft the falloff is,
 relative to the model's own units.
*>
fn float[] bind_geometric(
    alloc::Allocator allocator,
    Vec3[] positions,
    Vec3[] joints,
    int[] parents,
    float eps = 1.0e-4f)
{
    usz bones = joints.len;
    float[] dense = alloc::new_array(allocator, float, (sz)(positions.len * bones));

    foreach (usz v, position : positions)
    {
        usz base = v * bones;
        foreach (usz b, joint : joints)
        {
            int parent = parents[b];
            Vec3 head = parent >= 0 ? joints[(usz)parent] : joint;
            dense[base + b] = 1.0f / (segment_distance_sq(position, head, joint) + eps);
        }
    }

    return dense;
}

<*
 Write the mesh and its generated skeleton to a rigged .glb.

 Bone positions are mapped out of the encoder's normalised frame first, so the
 skeleton lands on the mesh in its own coordinates. Joints are named
 positionally — the checkpoint's skeleton templates carry real names, but
 matching a generated skeleton against one is a separate problem.

 @param geometry : `The mesh as it was handed to generate_rig, in its own coordinates.`
 @return? gltf::WEIGHT_SIZE_MISMATCH : `The skeleton and the weight matrix disagree.`
 @return? gltf::WRITE_FAILED : `The file could not be opened, written or flushed.`
*>
fn void? export_rig_glb(
    alloc::Allocator allocator,
    String path,
    TriMesh geometry,
    RigResult* rig,
    String name = "Mesh")
{
    Vec3[] joints = rig.joints_in_model_space(allocator);
    defer alloc::free(allocator, joints);

    gltf::Bone[] bones = alloc::new_array(allocator, gltf::Bone, (sz)joints.len);
    defer alloc::free(allocator, bones);

    foreach (i, joint : joints)
    {
        bones[i] = {
            .name = string::format(allocator, "joint_%d", i),
            .parent = rig.skeleton.parents[i],
            .position = joint,
        };
    }
    defer foreach (bone : bones) alloc::free(allocator, bone.name);

    float[] weights = bind_geometric(allocator, geometry.positions, joints, rig.skeleton.parents);
    defer alloc::free(allocator, weights);

    gltf::SkinnedMesh skinned = {
        .name = name,
        .positions = geometry.positions,
        .normals = geometry.normals,
        .indices = geometry.indices,
        .bones = bones,
        .weights = weights,
        .max_influences = 4,
    };

    return gltf::write_skinned_glb(path, skinned, allocator);
}
