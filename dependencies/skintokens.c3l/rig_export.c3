<*
 Writing a generated rig out as a .glb.

 glTF skinning wants a weight per bone per vertex. `SkinDecoder.decode_rig`
 produces those from the skin codes the transformer emitted after the
 skeleton, and `export_rig_glb` takes them as they come. When there are none —
 a sequence that ran out of budget partway through the skin block, or a caller
 that never asked for one — `bind_geometric` stands in with a plain distance
 falloff. That fallback is a placeholder, not a prediction: it puts a valid
 skin in the file and nothing more.

     `RigResult rig = skintokens::generate_rig(mem, &ctx, &encoder, &model, geometry)!;`
     `float[] skin = decoder.decode_rig(mem, &shape, &rig, geometry)!;`
     `skintokens::export_rig_glb(mem, "cow_rigged.glb", geometry, &rig, skin: skin)!;`
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
 The share of a mesh's vertices a joint has to lead to be worth keeping.

 A count would not travel: two vertices out of a 1300-vertex cow is noise, and
 two out of a 200000-vertex scan is nothing at all. A share means the same thing
 on both.

 One in four hundred sits in the middle of a wide gap on the cow — the weakest
 joint worth keeping leads 0.39% of the mesh, the weakest one worth dropping
 leads 0.15% — so the exact value is not doing delicate work. Raising it as far
 as 0.5% starts taking a fork that holds three chains together, which is where
 it stops being a cleanup.
*>
const float DEFAULT_PRUNE_SHARE = 0.0025f;

<*
 Drop the joints the skin decoder barely leans on, and the weight columns that
 went with them.

 The transformer emits a skeleton and then, separately, four codes per bone that
 the VAE turns into weights. The two halves can disagree, and when they do the
 weights are the half worth believing: a joint almost no vertex prefers does
 nothing in the rig but sit in the outliner and clutter the viewport. On the cow,
 five of thirty-three joints came back the strongest influence on not one vertex
 — three of them outside the mesh entirely — and a sixth led two.

 Dropping a joint hands its vertices to whatever they lean on next, which can
 lift another joint over the line — so they go one at a time, weakest first,
 recounting after each. It runs one round per joint removed and converges in the
 safe direction: removing a joint only ever adds to the counts of the ones that
 remain. The last joint is never taken, whatever the weights say.

 A pruned joint's children re-attach to its nearest surviving ancestor, so the
 tree stays connected and nothing is orphaned. Its weight column is dropped
 rather than handed to that ancestor; `sparsify_weights` renormalises what is
 left, which spreads the lost influence over the bones that were already sharing
 the vertex.

 @param skeleton : `Rewritten in place; its old arrays are freed.`
 @param skin : `[vertices, bones] as decode_rig returns it, replaced by the pruned matrix.`
 @param min_share : `Share of vertices a joint must lead. Zero keeps any joint that leads one.`
 @return `How many joints were removed.`
 @return? SKIN_SHAPE_MISMATCH : `skin is not exactly vertices * bones long.`
 @return? RIG_NO_BONES : `Every joint would go, which means the weights are empty.`
*>
fn usz? prune_unused_joints(
    alloc::Allocator allocator,
    Skeleton* skeleton,
    float[]* skin,
    usz vertices,
    float min_share = DEFAULT_PRUNE_SHARE)
{
    usz bones = skeleton.joints.len;
    if (bones == 0 || vertices == 0) return 0;
    if (skin.len != vertices * bones) return SKIN_SHAPE_MISMATCH~;

    // At least one vertex, however coarse the mesh: a share that rounds to
    // nothing would keep every joint, which is not what asking for a filter
    // means.
    //
    // The nudge before rounding up is not cosmetic. A float share widens to a
    // double a hair above itself — 0.2f is 0.20000000298 — so a bar that should
    // land exactly on a whole number of vertices lands a hair past it, and ceil
    // takes it to the next one. That turns "one vertex in five" into "one in
    // four" and quietly drops a joint that had cleared the bar.
    usz threshold = (usz)math::ceil((double)min_share * (double)vertices - 1.0e-6);
    if (threshold < 1) threshold = 1;

    bool[] keep = alloc::new_array(allocator, bool, (sz)bones);
    usz[] leads = alloc::new_array(allocator, usz, (sz)bones);
    defer {
        alloc::free(allocator, keep);
        alloc::free(allocator, leads);
    }
    for (usz b = 0; b < bones; b++) keep[b] = true;

    float[] weights = *skin;
    usz kept = bones;

    // The weakest joint at a time, never a whole pass at once. Two joints can
    // each lead one vertex and be under the bar together, and dropping either
    // one hands its vertex to the other and lifts it clear — taking both would
    // remove a joint that had earned its place.
    while (kept > 1)
    {
        for (usz b = 0; b < bones; b++) leads[b] = 0;

        for (usz v = 0; v < vertices; v++)
        {
            usz base = v * bones;
            usz best = bones;
            float strongest = 0;
            for (usz b = 0; b < bones; b++)
            {
                if (!keep[b]) continue;
                if (weights[base + b] > strongest) { strongest = weights[base + b]; best = b; }
            }
            if (best < bones) leads[best]++;
        }

        // Ties go to the later joint: the model emitted it further along, so it
        // is the more likely stray.
        usz weakest = bones;
        for (usz b = 0; b < bones; b++)
        {
            if (!keep[b]) continue;
            if (weakest == bones || leads[b] <= leads[weakest]) weakest = b;
        }
        if (weakest == bones || leads[weakest] >= threshold) break;

        keep[weakest] = false;
        kept--;
    }

    if (kept == bones) return 0;
    if (kept == 0) return RIG_NO_BONES~;

    // Old index to new. A pruned joint maps to the next survivor, which is what
    // a part marker sitting on it should move to.
    usz[] remap = alloc::new_array(allocator, usz, (sz)bones);
    defer alloc::free(allocator, remap);
    usz next = 0;
    for (usz b = 0; b < bones; b++)
    {
        remap[b] = next;
        if (keep[b]) next++;
    }

    Vec3[] joints = alloc::new_array(allocator, Vec3, (sz)kept);
    int[] parents = alloc::new_array(allocator, int, (sz)kept);
    for (usz b = 0; b < bones; b++)
    {
        if (!keep[b]) continue;
        joints[remap[b]] = skeleton.joints[b];

        // `resolve_parents` only ever attaches a bone to an earlier one, so
        // this walk terminates and every ancestor it lands on is already
        // renumbered.
        int ancestor = skeleton.parents[b];
        while (ancestor >= 0 && !keep[(usz)ancestor]) ancestor = skeleton.parents[(usz)ancestor];
        parents[remap[b]] = ancestor >= 0 ? (int)remap[(usz)ancestor] : -1;
    }

    float[] pruned = alloc::new_array(allocator, float, (sz)(vertices * kept));
    for (usz v = 0; v < vertices; v++)
    {
        usz from = v * bones;
        usz to = v * kept;
        for (usz b = 0; b < bones; b++)
        {
            if (keep[b]) pruned[to + remap[b]] = weights[from + b];
        }
    }

    foreach (&start : skeleton.part_starts)
    {
        *start = *start < bones ? remap[*start] : kept;
    }

    alloc::free(allocator, skeleton.joints);
    alloc::free(allocator, skeleton.parents);
    skeleton.joints = joints;
    skeleton.parents = parents;

    alloc::free(allocator, weights);
    *skin = pruned;

    return bones - kept;
}

<*
 Write the mesh and its generated skeleton to a rigged .glb.

 Bone positions are mapped out of the encoder's normalised frame first, so the
 skeleton lands on the mesh in its own coordinates. Joints are named
 positionally — the checkpoint's skeleton templates carry real names, but
 matching a generated skeleton against one is a separate problem.

 @param geometry : `The mesh as it was handed to generate_rig, in its own coordinates.`
 @param skin : `[vertices, bones] predicted weights, or empty for the distance fallback.`
 @return? gltf::WEIGHT_SIZE_MISMATCH : `The skeleton and the weight matrix disagree.`
 @return? gltf::WRITE_FAILED : `The file could not be opened, written or flushed.`
*>
fn void? export_rig_glb(
    alloc::Allocator allocator,
    String path,
    TriMesh geometry,
    RigResult* rig,
    String name = "Mesh",
    float[] skin = {})
{
    Vec3[] joints = rig.joints_in_model_space(allocator);
    defer alloc::free(allocator, joints);

    // Tails come out of the normalised frame the same way the joints do; a
    // direction is the difference of two points, so it has to be mapped as
    // points and subtracted after, not scaled on its own.
    Vec3[] tails = rig.skeleton.tails(allocator);
    defer alloc::free(allocator, tails);

    gltf::Bone[] bones = alloc::new_array(allocator, gltf::Bone, (sz)joints.len);
    defer alloc::free(allocator, bones);

    foreach (i, joint : joints)
    {
        bones[i] = {
            .name = string::format(allocator, "joint_%d", i),
            .parent = rig.skeleton.parents[i],
            .position = joint,
            .direction = rig.normalization.inverse(tails[i]) - joint,
        };
    }
    defer foreach (bone : bones) alloc::free(allocator, bone.name);

    bool predicted = skin.len == geometry.positions.len * joints.len && joints.len > 0;
    float[] fallback = predicted
        ? (float[]){}
        : bind_geometric(allocator, geometry.positions, joints, rig.skeleton.parents);
    defer if (!predicted) alloc::free(allocator, fallback);

    gltf::SkinnedMesh skinned = {
        .name = name,
        .positions = geometry.positions,
        .normals = geometry.normals,
        .indices = geometry.indices,
        .bones = bones,
        .weights = predicted ? skin : fallback,
        .max_influences = 4,
    };

    return gltf::write_skinned_glb(path, skinned, allocator);
}
