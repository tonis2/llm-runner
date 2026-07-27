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
