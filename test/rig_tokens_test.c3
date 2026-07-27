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
