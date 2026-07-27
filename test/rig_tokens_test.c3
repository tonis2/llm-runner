module llm;

import skintokens;
import std::io;
import std::core::mem;
import std::core::test;
import std::math;
import std::math::vector;

fn bool near(float a, float b, float epsilon = 1e-5f) => math::abs(a - b) <= epsilon;

fn void expect_joint(skintokens::Skeleton* s, usz bone, float x, float y, float z)
{
    Vec3 j = s.joints[bone];
    assert(near(j[0], x) && near(j[1], y) && near(j[2], z),
        "bone %d at (%f, %f, %f), expected (%f, %f, %f)", bone, j[0], j[1], j[2], x, y, z);
}

fn void test_coordinate_bins_round_trip() @test
{
    assert(skintokens::NUM_DISCRETE == 256);
    assert(skintokens::TOKEN_SEQUENCE_END == 33035, "the model's own terminator");
    assert(skintokens::RIG_TOTAL_VOCAB == 33036, "matches the embedding table");

    // Bin centres: bin 0 is just inside -1, bin 255 just inside +1, and 128
    // straddles the origin.
    assert(near(skintokens::undiscretize(0), -1.0f + 1.0f / 256.0f));
    assert(near(skintokens::undiscretize(255), 1.0f - 1.0f / 256.0f));
    assert(near(skintokens::undiscretize(128), 1.0f / 256.0f));

    // Bins are evenly spaced, one 128th apart.
    for (uint bin = 1; bin < 256; bin++)
    {
        assert(near(skintokens::undiscretize(bin) - skintokens::undiscretize(bin - 1), 1.0f / 128.0f),
            "bin %d is not one step from its neighbour", bin);
    }

    // discretize centres its bins on integers of (x + 1) * 128, so feeding it
    // one lands exactly on that bin.
    for (uint bin = 1; bin < 255; bin++)
    {
        float on_centre = (float)bin / 128.0f - 1.0f;
        assert(skintokens::discretize(on_centre) == bin, "bin %d centre landed elsewhere", bin);
    }

    // The two are not inverses, and the gap is worth pinning rather than
    // discovering later: undiscretize places a value half a bin above what
    // discretize considers that bin's centre, so decoding then re-encoding
    // lands on a rounding tie every time. Half-to-even then sends even bins
    // back to themselves and odd bins one higher. Both halves are the
    // reference's, so this offset is the representation, not a defect to fix.
    for (uint bin = 0; bin < 255; bin++)
    {
        uint again = skintokens::discretize(skintokens::undiscretize(bin));
        uint want = (bin % 2 == 0) ? bin : bin + 1;
        assert(again == want, "bin %d re-encoded to %d, expected %d", bin, again, want);
    }

    // And anything off the ends clamps rather than wrapping.
    assert(skintokens::discretize(-9.0f) == 0);
    assert(skintokens::discretize(9.0f) == 255);
}

// A chain: root, then two bones continuing from it. Three tokens each, and
// each one hangs off the joint before it.
fn void test_detokenize_a_simple_chain() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS,
        skintokens::TOKEN_CLS_VROID,
        128, 128, 128,
        128, 200, 128,
        128, 255, 128,
        skintokens::TOKEN_EOS,
    };

    skintokens::Skeleton skeleton = skintokens::detokenize_skeleton(mem, ids)!!;
    defer skeleton.free(mem);

    assert(skeleton.len() == 3);
    assert(skeleton.rig_class == skintokens::RigClass.VROID);

    expect_joint(&skeleton, 0, skintokens::undiscretize(128), skintokens::undiscretize(128), skintokens::undiscretize(128));
    expect_joint(&skeleton, 1, skintokens::undiscretize(128), skintokens::undiscretize(200), skintokens::undiscretize(128));

    assert(skeleton.parents[0] == -1, "the first bone is the root");
    assert(skeleton.parents[1] == 0);
    assert(skeleton.parents[2] == 1);
}

// A branch restarts the chain somewhere else, and spells out where it hangs
// from before where it sits — six coordinates rather than three.
fn void test_detokenize_a_branch() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS,
        skintokens::TOKEN_CLS_NONE,
        100, 100, 100,          // bone 0, root
        100, 150, 100,          // bone 1, continues from 0
        skintokens::TOKEN_BRANCH,
        100, 100, 100,          // bone 2 hangs off bone 0's joint
        60, 120, 100,
        skintokens::TOKEN_EOS,
    };

    skintokens::Skeleton skeleton = skintokens::detokenize_skeleton(mem, ids)!!;
    defer skeleton.free(mem);

    assert(skeleton.len() == 3);
    assert(skeleton.parents[0] == -1);
    assert(skeleton.parents[1] == 0);
    // Resolved by position: the branch named bone 0's joint, not bone 1's.
    assert(skeleton.parents[2] == 0, "branch attached to bone %d, expected 0", skeleton.parents[2]);
    expect_joint(&skeleton, 2, skintokens::undiscretize(60), skintokens::undiscretize(120), skintokens::undiscretize(100));
}

// Rigs really do have coincident joints — a shoulder and an upper arm on the
// same point. The reference scans backwards and keeps the first strict
// improvement, so the most recent joint wins; picking the other one silently
// reparents a limb.
fn void test_coincident_joints_take_the_most_recent_parent() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS,
        skintokens::TOKEN_CLS_NONE,
        100, 100, 100,          // bone 0
        100, 100, 100,          // bone 1, same position
        skintokens::TOKEN_BRANCH,
        100, 100, 100,          // ambiguous: matches both 0 and 1
        50, 50, 50,
        skintokens::TOKEN_EOS,
    };

    skintokens::Skeleton skeleton = skintokens::detokenize_skeleton(mem, ids)!!;
    defer skeleton.free(mem);

    assert(skeleton.parents[2] == 1,
        "a tie should pick the most recent joint, picked %d", skeleton.parents[2]);
}

fn void test_part_tokens_are_recorded() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS,
        skintokens::TOKEN_CLS_VROID,
        skintokens::TOKEN_PART_BODY,
        128, 128, 128,
        128, 140, 128,
        skintokens::TOKEN_PART_HAND,
        128, 150, 128,
        skintokens::TOKEN_EOS,
    };

    skintokens::Skeleton skeleton = skintokens::detokenize_skeleton(mem, ids)!!;
    defer skeleton.free(mem);

    assert(skeleton.len() == 3);
    assert(skeleton.part_starts.len == 2);
    // The body part opens before any bone, the hand part after two.
    assert(skeleton.part_starts[0] == 0 && skeleton.part_tokens[0] == skintokens::TOKEN_PART_BODY);
    assert(skeleton.part_starts[1] == 2 && skeleton.part_tokens[1] == skintokens::TOKEN_PART_HAND);
}

fn void test_detokenize_rejects_malformed_sequences() @test
{
    uint[] no_bos = { 128, 128, 128, skintokens::TOKEN_EOS };
    assert(@catch(skintokens::detokenize_skeleton(mem, no_bos)) == skintokens::RIG_NOT_BOS);

    uint[] unterminated = { skintokens::TOKEN_BOS, 128, 128, 128 };
    assert(@catch(skintokens::detokenize_skeleton(mem, unterminated)) == skintokens::RIG_NOT_TERMINATED);

    uint[] truncated = { skintokens::TOKEN_BOS, 128, 128, skintokens::TOKEN_EOS };
    assert(@catch(skintokens::detokenize_skeleton(mem, truncated)) == skintokens::RIG_TRUNCATED_COORDINATES);

    uint[] empty = { skintokens::TOKEN_BOS, skintokens::TOKEN_EOS };
    assert(@catch(skintokens::detokenize_skeleton(mem, empty)) == skintokens::RIG_NO_BONES);

    // A skin token in a skeleton means the sequence is being read at the wrong
    // offset, which is worth failing loudly rather than skipping.
    uint[] skin = { skintokens::TOKEN_BOS, 128, 128, 128, skintokens::SKIN_TOKEN_BASE + 5, skintokens::TOKEN_EOS };
    assert(@catch(skintokens::detokenize_skeleton(mem, skin)) == skintokens::RIG_UNEXPECTED_TOKEN);
}

// The model's own terminator ends a sequence too: greedy decoding can reach
// for 33035 rather than the tokenizer's eos.
fn void test_sequence_end_terminates() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_NONE,
        128, 128, 128,
        skintokens::TOKEN_SEQUENCE_END,
    };

    skintokens::Skeleton skeleton = skintokens::detokenize_skeleton(mem, ids)!!;
    defer skeleton.free(mem);
    assert(skeleton.len() == 1);
    assert(skeleton.parents[0] == -1);
}

// --- The skin phase -------------------------------------------------------

fn skintokens::RigGrammar run_grammar(uint[] tokens, uint tokens_per_skin)
{
    skintokens::RigGrammar grammar = { .tokens_per_skin = tokens_per_skin };
    foreach (token : tokens) grammar.accept(token);
    return grammar;
}

fn bool allows(skintokens::RigGrammar* grammar, uint token)
{
    bool[] allowed = mem::new_array(bool, (sz)skintokens::RIG_TOTAL_VOCAB);
    defer mem::free(allowed);
    grammar.allow_next(allowed);
    return allowed[token];
}

// The grammar counts bones the way the reference's `bones_in_sequence` does,
// because that count is what decides how many skin codes the model owes: a
// branch spends six coordinates naming a parent and then a joint, and still
// closes only one bone.
fn void test_grammar_counts_bones_like_the_tokenizer() @test
{
    uint[] chain = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_ARTICULATION,
        10, 20, 30,
        40, 50, 60,
    };
    skintokens::RigGrammar g = run_grammar(chain, 0);
    assert(g.bones == 2, "two joints in a chain are two bones, got %d", g.bones);

    uint[] branched = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_ARTICULATION,
        10, 20, 30,
        skintokens::TOKEN_BRANCH,
        10, 20, 30,      // where it hangs from
        70, 80, 90,      // where it sits
    };
    skintokens::RigGrammar b = run_grammar(branched, 0);
    assert(b.bones == 2, "a branch adds one bone, not two, got %d", b.bones);

    // And the count matches what detokenize_skeleton actually builds.
    uint[] with_eos = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_ARTICULATION,
        10, 20, 30,
        skintokens::TOKEN_BRANCH,
        10, 20, 30,
        70, 80, 90,
        skintokens::TOKEN_EOS,
    };
    skintokens::Skeleton skeleton = skintokens::detokenize_skeleton(mem, with_eos)!!;
    defer skeleton.free(mem);
    assert(skeleton.len() == (usz)b.bones, "grammar and detokenizer disagree on the bone count");
}

// Zero tokens_per_skin makes the skeleton's terminator final. That is what a
// caller wanting bones and nothing else asks for, and it is what the seed
// search in examples/rig runs with.
fn void test_grammar_stops_at_the_skeleton_when_skin_is_off() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_ARTICULATION,
        10, 20, 30,
        skintokens::TOKEN_EOS,
    };
    skintokens::RigGrammar g = run_grammar(ids, 0);
    assert(g.finished(), "eos should be terminal with no skin block asked for");
}

fn void test_grammar_counts_out_the_skin_block() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_ARTICULATION,
        10, 20, 30,
        40, 50, 60,
        skintokens::TOKEN_EOS,
    };
    skintokens::RigGrammar g = run_grammar(ids, skintokens::TOKENS_PER_SKIN);

    assert(!g.finished(), "the skin block has not started yet");
    assert(g.bones == 2);
    assert(g.skin_remaining() == 2 * skintokens::TOKENS_PER_SKIN);

    // Only skin codes are reachable here. The reference reopens everything
    // from the skeleton's eos upward; training puts nothing but codes in this
    // stretch, so the tighter mask is the correct one.
    assert(allows(&g, skintokens::SKIN_TOKEN_BASE));
    assert(allows(&g, skintokens::TOKEN_SEQUENCE_END - 1));
    assert(!allows(&g, skintokens::TOKEN_SEQUENCE_END), "not until the block is full");
    assert(!allows(&g, skintokens::TOKEN_PAD));
    assert(!allows(&g, 128), "coordinates are done");

    for (uint i = 0; i < 2 * skintokens::TOKENS_PER_SKIN; i++)
    {
        assert(!g.finished());
        g.accept(skintokens::SKIN_TOKEN_BASE + i);
    }

    assert(g.skin_remaining() == 0);
    assert(allows(&g, skintokens::TOKEN_SEQUENCE_END), "the block is full, so the terminator is due");
    assert(!allows(&g, skintokens::SKIN_TOKEN_BASE), "and nothing else is");

    g.accept(skintokens::TOKEN_SEQUENCE_END);
    assert(g.finished());
}

// The reference stops one code short: its logits processor compares
// `length - where` against `J * tokens_per_skin`, and at the step just after
// the skeleton's eos that difference is already 1. So it forces the model's
// terminator after J*4-1 codes, and `decode` then slices J*4 values — the last
// of which is the terminator itself, 33035, which unpacks to codebook entry 0.
// The last bone's fourth latent is silently wrong there. Training writes the
// full J*4 followed by the terminator, so counting the whole block is both
// what the model was taught and what leaves every bone intact.
fn void test_skin_block_is_a_full_four_per_bone() @test
{
    uint[] ids = {
        skintokens::TOKEN_BOS, skintokens::TOKEN_CLS_ARTICULATION,
        10, 20, 30,
        skintokens::TOKEN_EOS,
    };
    skintokens::RigGrammar g = run_grammar(ids, skintokens::TOKENS_PER_SKIN);
    assert(g.bones == 1);

    uint emitted = 0;
    while (!allows(&g, skintokens::TOKEN_SEQUENCE_END))
    {
        g.accept(skintokens::SKIN_TOKEN_BASE + emitted);
        emitted++;
        assert(emitted <= 8, "the block should have closed by now");
    }
    assert(emitted == skintokens::TOKENS_PER_SKIN,
        "one bone owes four codes, not %d", emitted);
}

// --- Bone directions ------------------------------------------------------

fn skintokens::Skeleton make_skeleton(Vec3[] joints, int[] parents)
{
    skintokens::Skeleton s;
    s.joints = mem::new_array(Vec3, (sz)joints.len);
    s.parents = mem::new_array(int, (sz)parents.len);
    foreach (i, j : joints) s.joints[i] = j;
    foreach (i, p : parents) s.parents[i] = p;
    return s;
}

fn Vec3[] tails_of(skintokens::Skeleton* skeleton) => skeleton.tails(mem);

// A fork follows the child the model emitted first — the limb it was already
// walking down. Every later child arrived through a `branch` token, which names
// a position to hang off rather than a step to take, so it can be anywhere.
fn void test_a_fork_follows_the_chain_it_was_walking() @test
{
    // 0 -> 1 -> 2 is the walk; joint 3 then branches back onto 0 and runs on to
    // 6, so it is both deeper and, here, no further away.
    Vec3[7] joints = {
        { 0,  0, 0 },
        { 0, -1, 0 },
        { 0, -2, 0 },
        { 1,  0, 0 },
        { 2,  0, 0 },
        { 3,  0, 0 },
        { 4,  0, 0 },
    };
    int[7] parents = { -1, 0, 1, 0, 3, 4, 5 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);
    Vec3[] tails = tails_of(&skeleton);
    defer mem::free(tails);

    assert(tails[0] == joints[1], "the fork left the chain it was walking for a branch");
    assert(tails[1] == joints[2], "a single child is still just followed");

    // Every bone with a child reaches it exactly, which is what keeps a
    // skeleton from breaking into stubs at each fork.
    foreach (usz i, tail : tails)
    {
        bool has_child = false;
        foreach (parent : skeleton.parents) if (parent >= 0 && (usz)parent == i) has_child = true;
        if (!has_child) continue;

        bool lands_on_a_joint = false;
        foreach (usz j, joint : skeleton.joints) if (j != i && tail == joint) lands_on_a_joint = true;
        assert(lands_on_a_joint, "bone %d ends nowhere", i);
    }
}

// The two rules that sound better both fail on a real rig, so pin what they
// would have done here. A branch re-attaches to an arbitrary earlier joint, so
// it can be far away and can still be the longest chain hanging off the fork.
fn void test_a_fork_ignores_a_deeper_or_nearer_branch() @test
{
    // 0 -> 1 -> 2 is the walk. Joint 3 branches onto 0 much further out and
    // runs four joints, so it is the deepest chain; joint 7 branches onto 0
    // closer than joint 1, so it is the nearest child.
    Vec3[8] joints = {
        { 0,   0, 0 },
        { 0,  -1, 0 },
        { 0,  -2, 0 },
        { 9,   0, 0 },
        { 10,  0, 0 },
        { 11,  0, 0 },
        { 12,  0, 0 },
        { 0.1, 0, 0 },
    };
    int[8] parents = { -1, 0, 1, 0, 3, 4, 5, 0 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);
    Vec3[] tails = tails_of(&skeleton);
    defer mem::free(tails);

    assert(tails[0] != joints[3], "aimed at the deepest chain and crossed the model to do it");
    assert(tails[0] != joints[7], "aimed at the nearest child instead of the limb");
    assert(tails[0] == joints[1], "should follow the chain it was walking");
}

// A leaf has nothing to reach, so it carries on the way it came — half as far
// again. That much is the reference's rule and stays.
fn void test_a_leaf_extrudes_the_way_it_came() @test
{
    Vec3[3] joints = { { 0, 0, 0 }, { 0, -1, 0 }, { 0, -3, 0 } };
    int[3] parents = { -1, 0, 1 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);
    Vec3[] tails = tails_of(&skeleton);
    defer mem::free(tails);

    // The last bone arrives along (0, -2, 0), so it ends one further unit on.
    assert(tails[2] == (Vec3){ 0, -4, 0 }, "leaf extrusion is half the arriving bone");

    // A lone joint has neither a parent nor a child; it goes up rather than
    // collapsing to zero length, which would leave it undrawable.
    Vec3[1] alone = { { 5, 5, 5 } };
    int[1] no_parent = { -1 };
    skintokens::Skeleton single = make_skeleton(alone[..], no_parent[..]);
    defer single.free(mem);
    Vec3[] one = tails_of(&single);
    defer mem::free(one);
    assert(one[0].y > alone[0].y, "a lone joint needs some length");
}

// --- Pruning --------------------------------------------------------------

// The transformer's two halves can disagree, and the weights are the half
// worth believing: a joint no vertex leans on does nothing but clutter the rig.
fn void test_pruning_drops_joints_no_vertex_leans_on() @test
{
    // 0 -> 1 -> 2 is real; 3 hangs off 1 and 4 hangs off 3, and neither is any
    // vertex's strongest, so 4 has to survive 3 going by re-attaching to 1.
    Vec3[5] joints = {
        { 0, 0, 0 }, { 0, -1, 0 }, { 0, -2, 0 }, { 1, -1, 0 }, { 2, -1, 0 },
    };
    int[5] parents = { -1, 0, 1, 1, 3 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);

    // Three vertices, each led by one of 0, 1, 2. Joints 3 and 4 pick up a
    // little weight everywhere but lead nothing.
    float[15] dense = {
        0.8f, 0.1f, 0.0f, 0.05f, 0.05f,
        0.1f, 0.7f, 0.1f, 0.05f, 0.05f,
        0.0f, 0.2f, 0.7f, 0.05f, 0.05f,
    };
    // One defer, after the call: prune_unused_joints frees the array it was
    // given and hands back a new one, and a defer reads `skin` at scope exit.
    float[] skin = mem::new_array(float, 15);
    foreach (i, w : dense) skin[i] = w;

    usz dropped = skintokens::prune_unused_joints(mem, &skeleton, &skin, 3)!!;
    defer mem::free(skin);

    assert(dropped == 2, "expected joints 3 and 4 to go, dropped %d", dropped);
    assert(skeleton.len() == 3);
    assert(skin.len == 3 * 3, "the weight matrix should have lost two columns");

    // The survivors keep their order, their positions and their tree.
    assert(skeleton.joints[0] == joints[0] && skeleton.joints[1] == joints[1]
        && skeleton.joints[2] == joints[2]);
    assert(skeleton.parents[0] == -1 && skeleton.parents[1] == 0 && skeleton.parents[2] == 1);

    // And the columns that stayed are the ones that were there before.
    assert(skin[0] == 0.8f && skin[1] == 0.1f && skin[2] == 0.0f);
    assert(skin[6] == 0.0f && skin[7] == 0.2f && skin[8] == 0.7f);
}

// A pruned joint's children re-attach to the nearest surviving ancestor rather
// than being orphaned, even when several ancestors go at once.
fn void test_pruning_reattaches_orphans() @test
{
    // A chain 0 -> 1 -> 2 -> 3 where only the ends are used, so 3 has to climb
    // past two dead joints to find 0.
    Vec3[4] joints = { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 2, 0 }, { 0, 3, 0 } };
    int[4] parents = { -1, 0, 1, 2 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);

    float[8] dense = {
        0.9f, 0.03f, 0.03f, 0.04f,
        0.1f, 0.05f, 0.05f, 0.8f,
    };
    float[] skin = mem::new_array(float, 8);
    foreach (i, w : dense) skin[i] = w;

    usz dropped = skintokens::prune_unused_joints(mem, &skeleton, &skin, 2)!!;
    defer mem::free(skin);

    assert(dropped == 2);
    assert(skeleton.len() == 2);
    assert(skeleton.parents[0] == -1);
    assert(skeleton.parents[1] == 0, "the orphan should have climbed to the root");
    assert(skeleton.joints[1] == joints[3]);
}

// Nothing to do is not an error, and it must not disturb what is already there.
fn void test_pruning_a_fully_used_skeleton_changes_nothing() @test
{
    Vec3[2] joints = { { 0, 0, 0 }, { 0, 1, 0 } };
    int[2] parents = { -1, 0 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);

    float[] skin = mem::new_array(float, 4);
    skin[0] = 0.9f; skin[1] = 0.1f;
    skin[2] = 0.2f; skin[3] = 0.8f;

    usz dropped = skintokens::prune_unused_joints(mem, &skeleton, &skin, 2)!!;
    defer mem::free(skin);

    assert(dropped == 0);
    assert(skeleton.len() == 2);
    assert(skin.len == 4);
}

// A joint that leads a couple of vertices out of a large mesh is noise, so the
// filter is a share rather than a count — two vertices mean something different
// on a 1300-vertex cow and on a 200000-vertex scan.
fn void test_pruning_measures_a_share_not_a_count() @test
{
    // Four joints, ten vertices. Joint 3 leads exactly one of them, which is
    // 10% -- above a 5% bar and below a 20% one.
    Vec3[4] joints = { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 2, 0 }, { 1, 1, 0 } };
    int[4] parents = { -1, 0, 1, 1 };

    float[] dense = mem::new_array(float, 40);
    defer mem::free(dense);
    for (usz v = 0; v < 10; v++)
    {
        usz lead = v < 4 ? 0 : (v < 7 ? 1 : 2);
        if (v == 9) lead = 3;
        for (usz b = 0; b < 4; b++) dense[v * 4 + b] = (b == lead) ? 0.7f : 0.1f;
    }

    // 5%: one vertex in ten clears the bar, so joint 3 stays.
    {
        skintokens::Skeleton keep = make_skeleton(joints[..], parents[..]);
        defer keep.free(mem);
        float[] skin = mem::new_array(float, 40);
        foreach (i, w : dense) skin[i] = w;
        usz dropped = skintokens::prune_unused_joints(mem, &keep, &skin, 10, 0.05f)!!;
        defer mem::free(skin);
        assert(dropped == 0, "a joint leading 10%% should clear a 5%% bar");
        assert(keep.len() == 4);
    }

    // 20%: it does not, and its one vertex falls to whatever it leans on next.
    {
        skintokens::Skeleton cut = make_skeleton(joints[..], parents[..]);
        defer cut.free(mem);
        float[] skin = mem::new_array(float, 40);
        foreach (i, w : dense) skin[i] = w;
        usz dropped = skintokens::prune_unused_joints(mem, &cut, &skin, 10, 0.20f)!!;
        defer mem::free(skin);
        assert(dropped == 1, "a joint leading 10%% should fail a 20%% bar, dropped %d", dropped);
        assert(cut.len() == 3);
        assert(skin.len == 10 * 3);
    }
}

// Dropping a joint hands its vertices on, which can lift another over the bar,
// so the filter repeats until a pass finds nothing. A single pass would take
// both of these.
fn void test_pruning_repeats_until_nothing_falls() @test
{
    Vec3[3] joints = { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 2, 0 } };
    int[3] parents = { -1, 0, 1 };

    skintokens::Skeleton skeleton = make_skeleton(joints[..], parents[..]);
    defer skeleton.free(mem);

    // Ten vertices. Joint 0 leads eight. Joints 1 and 2 lead one each, and both
    // of joint 2's would go to joint 1 if joint 2 were dropped -- lifting it to
    // two and over a 20% bar.
    float[] skin = mem::new_array(float, 30);
    for (usz v = 0; v < 10; v++)
    {
        usz base = v * 3;
        if (v < 8)      { skin[base] = 0.8f; skin[base+1] = 0.1f; skin[base+2] = 0.1f; }
        else if (v == 8){ skin[base] = 0.1f; skin[base+1] = 0.8f; skin[base+2] = 0.1f; }
        else            { skin[base] = 0.1f; skin[base+1] = 0.3f; skin[base+2] = 0.6f; }
    }

    usz dropped = skintokens::prune_unused_joints(mem, &skeleton, &skin, 10, 0.20f)!!;
    defer mem::free(skin);

    assert(dropped == 1, "only joint 2 should go; joint 1 inherits its vertex, dropped %d", dropped);
    assert(skeleton.len() == 2);
    assert(skeleton.joints[1] == joints[1], "the joint that was rescued is the one that stayed");
}
