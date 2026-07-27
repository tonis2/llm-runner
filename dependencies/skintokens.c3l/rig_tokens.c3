<*
 The rig token grammar: what the transformer emits, and how it becomes bones.

 The vocabulary is 33036 wide and splits into three parts:

     0     .. 255     coordinate bins, 256 of them over [-1, 1]
     256   .. 266     the specials below
     267   .. 33034   skin tokens, an FSQ code offset by the 267 above
     33035            the sequence terminator

 A skeleton is spelled as a walk down the bone chain. Most bones give three
 tokens — the coordinates of their joint — and inherit their parent from
 whatever came before. A `branch` token means the chain restarts somewhere
 else, and is followed by six coordinates: where the new bone hangs from, then
 where its own joint sits. Part tokens mark where the body ends and the hands
 begin, which is what lets the joint names be recovered from a template.

 Nothing here touches the GPU. It is the half of generation that can be checked
 exactly, so it is worth keeping separable from the half that cannot.
*>
module skintokens;

import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::vector;

faultdef
    RIG_NOT_BOS,
    RIG_NOT_TERMINATED,
    RIG_UNEXPECTED_TOKEN,
    RIG_TRUNCATED_COORDINATES,
    RIG_NO_BONES;

// From the checkpoint's tokenizer_config: num_discrete 256 over [-1, 1], then
// branch/bos/eos/pad, spring, the parts (body, hand), cls_none, and the
// classes (rignet, vroid, articulation) — in that order, which is the order
// TokenizerPart.parse assigns them.
const uint NUM_DISCRETE = 256;
const uint TOKEN_BRANCH = 256;
const uint TOKEN_BOS = 257;
const uint TOKEN_EOS = 258;
const uint TOKEN_PAD = 259;
const uint TOKEN_SPRING = 260;
const uint TOKEN_PART_BODY = 261;
const uint TOKEN_PART_HAND = 262;
const uint TOKEN_CLS_NONE = 263;
const uint TOKEN_CLS_RIGNET = 264;
const uint TOKEN_CLS_VROID = 265;
const uint TOKEN_CLS_ARTICULATION = 266;

// Where the tokenizer's own vocabulary ends and the FSQ skin codes begin.
const uint RIG_VOCAB_SIZE = 267;
const uint SKIN_TOKEN_BASE = RIG_VOCAB_SIZE;
const uint SKIN_TOKEN_COUNT = 32768;

// The model's own end-of-sequence, distinct from the tokenizer's eos: the
// transformer's vocabulary is the tokenizer's plus the skin codes plus this.
const uint TOKEN_SEQUENCE_END = SKIN_TOKEN_BASE + SKIN_TOKEN_COUNT;  // 33035
const uint RIG_TOTAL_VOCAB = TOKEN_SEQUENCE_END + 1;                 // 33036

enum RigClass {
    NONE,
    RIGNET,
    VROID,
    ARTICULATION,
}

fn uint RigClass.token(self) {
    switch (self) {
        case RIGNET: return TOKEN_CLS_RIGNET;
        case VROID: return TOKEN_CLS_VROID;
        case ARTICULATION: return TOKEN_CLS_ARTICULATION;
        default: return TOKEN_CLS_NONE;
    }
}

fn bool is_coordinate(uint token) => token < NUM_DISCRETE;
fn bool is_skin_token(uint token) => token >= SKIN_TOKEN_BASE && token < TOKEN_SEQUENCE_END;

<*
 Bin index back to a coordinate, at the centre of its bin.

 The inverse of the tokenizer's `discretize` over [-1, 1], so a bin is
 2/256 = 0.0078 wide and the reconstruction carries half that as error. Bone
 positions are only ever this precise; that is the representation, not a bug.
*>
fn float undiscretize(uint bin) {
    return ((float)bin + 0.5f) / (float)NUM_DISCRETE * 2.0f - 1.0f;
}

<*
 A coordinate to its bin.

 Ties go to even, matching numpy's `.round()`, and that is not academic here:
 a bin centre maps to exactly `bin + 0.5` in this scale, so every value
 `undiscretize` produces is a tie. Round-half-up would push all of them one bin
 high, which is a systematic drift rather than a rounding difference.
*>
fn uint discretize(float value) {
    double scaled = ((double)value + 1.0) * 0.5 * (double)NUM_DISCRETE;
    double lower = math::floor(scaled);
    double frac = scaled - lower;

    long bin = (long)lower;
    if (frac > 0.5) {
        bin += 1;
    } else if (frac == 0.5 && (bin & 1) != 0) {
        bin += 1;
    }

    if (bin < 0) return 0;
    if (bin >= (long)NUM_DISCRETE) return NUM_DISCRETE - 1;
    return (uint)bin;
}

<*
 A skeleton as the tokenizer describes it: a joint per bone, and the index of
 the bone it hangs from.

 Positions are in the normalised frame the encoder saw, so
 `Normalization.inverse` is what puts them back on the mesh.
*>
struct Skeleton {
    Vec3[] joints;
    int[] parents;
    RigClass rig_class;
    // Bone index where each part token appeared, and which part it was, so a
    // template can name the joints. Empty when the model emitted none.
    usz[] part_starts;
    uint[] part_tokens;
}

fn usz Skeleton.len(self) => self.joints.len;

fn void Skeleton.free(&self, Allocator allocator) {
    if (self.joints.len > 0) alloc::free(allocator, self.joints);
    if (self.parents.len > 0) alloc::free(allocator, self.parents);
    if (self.part_starts.len > 0) alloc::free(allocator, self.part_starts);
    if (self.part_tokens.len > 0) alloc::free(allocator, self.part_tokens);
    *self = {};
}

// Growing lists without pulling in std::collections for four small arrays.
struct JointBuffer @local {
    Vec3[] joints;
    Vec3[] parent_joints;
    usz count;
}

fn void JointBuffer.push(&self, Vec3 joint, Vec3 parent_joint) @local {
    self.joints[self.count] = joint;
    self.parent_joints[self.count] = parent_joint;
    self.count++;
}

<*
 Turn a generated token sequence into bones.

 `ids` is the whole sequence including the leading bos; trailing pad is
 tolerated, and everything from the tokenizer's eos onward is ignored. Skin
 tokens are not part of a skeleton and are rejected rather than skipped — a
 sequence carrying them is being read at the wrong offset.

 @param ids : `The token sequence, starting at bos.`
 @return? RIG_NOT_BOS : `The sequence does not start with bos.`
 @return? RIG_NOT_TERMINATED : `No eos before the end of the sequence.`
 @return? RIG_UNEXPECTED_TOKEN : `A token that cannot appear in a skeleton.`
 @return? RIG_TRUNCATED_COORDINATES : `A coordinate triple ran off the end.`
 @return? RIG_NO_BONES : `The sequence was well formed but described nothing.`
*>
fn Skeleton? detokenize_skeleton(Allocator allocator, uint[] ids) {
    if (ids.len == 0 || ids[0] != TOKEN_BOS) return RIG_NOT_BOS~;

    // Everything up to the tokenizer's eos. The model's own terminator counts
    // too: greedy decoding can reach for it instead.
    usz end = 0;
    bool terminated = false;
    for (usz i = 1; i < ids.len; i++) {
        if (ids[i] == TOKEN_EOS || ids[i] == TOKEN_SEQUENCE_END) {
            end = i;
            terminated = true;
            break;
        }
    }
    if (!terminated) return RIG_NOT_TERMINATED~;

    uint[] body = ids[1..end - 1];
    if (body.len == 0) return RIG_NO_BONES~;

    // Three tokens is the shortest a bone can be, so this is a safe ceiling.
    usz capacity = body.len / 3 + 1;
    JointBuffer buffer = {
        .joints = alloc::new_array(allocator, Vec3, (sz)capacity),
        .parent_joints = alloc::new_array(allocator, Vec3, (sz)capacity),
    };
    defer {
        alloc::free(allocator, buffer.joints);
        alloc::free(allocator, buffer.parent_joints);
    }

    usz[] part_starts = alloc::new_array(allocator, usz, (sz)(body.len + 1));
    uint[] part_tokens = alloc::new_array(allocator, uint, (sz)(body.len + 1));
    defer {
        alloc::free(allocator, part_starts);
        alloc::free(allocator, part_tokens);
    }
    usz part_count = 0;

    RigClass rig_class = NONE;
    bool branching = false;
    bool have_last = false;
    Vec3 last_joint;

    usz i = 0;
    while (i < body.len) {
        uint token = body[i];

        if (is_coordinate(token)) {
            Vec3 joint;
            Vec3 parent_joint;

            if (branching) {
                // A branch names where it hangs from before where it sits.
                if (i + 6 > body.len) return RIG_TRUNCATED_COORDINATES~;
                parent_joint = { undiscretize(body[i]), undiscretize(body[i + 1]), undiscretize(body[i + 2]) };
                joint = { undiscretize(body[i + 3]), undiscretize(body[i + 4]), undiscretize(body[i + 5]) };
                i += 6;
            } else {
                if (i + 3 > body.len) return RIG_TRUNCATED_COORDINATES~;
                joint = { undiscretize(body[i]), undiscretize(body[i + 1]), undiscretize(body[i + 2]) };
                // Continuing a chain: this bone hangs off the previous joint.
                // The very first bone is the root and hangs off itself.
                parent_joint = have_last ? last_joint : joint;
                i += 3;
            }

            for (usz c = 0; c < 3; c++) {
                if (!is_coordinate(body[i - 3 + c])) return RIG_UNEXPECTED_TOKEN~;
            }

            buffer.push(joint, parent_joint);
            last_joint = joint;
            have_last = true;
            branching = false;
            continue;
        }

        switch (token) {
            case TOKEN_BRANCH:
                branching = true;
                have_last = false;
            case TOKEN_SPRING:
            case TOKEN_PART_BODY:
            case TOKEN_PART_HAND:
                part_starts[part_count] = buffer.count;
                part_tokens[part_count] = token;
                part_count++;
            case TOKEN_CLS_NONE:
                rig_class = NONE;
            case TOKEN_CLS_RIGNET:
                rig_class = RIGNET;
            case TOKEN_CLS_VROID:
                rig_class = VROID;
            case TOKEN_CLS_ARTICULATION:
                rig_class = ARTICULATION;
            default:
                return RIG_UNEXPECTED_TOKEN~;
        }
        i++;
    }

    if (buffer.count == 0) return RIG_NO_BONES~;

    Skeleton skeleton = {
        .joints = alloc::new_array(allocator, Vec3, (sz)buffer.count),
        .parents = alloc::new_array(allocator, int, (sz)buffer.count),
        .rig_class = rig_class,
        .part_starts = alloc::new_array(allocator, usz, (sz)part_count),
        .part_tokens = alloc::new_array(allocator, uint, (sz)part_count),
    };
    for (usz b = 0; b < buffer.count; b++) skeleton.joints[b] = buffer.joints[b];
    for (usz p = 0; p < part_count; p++) {
        skeleton.part_starts[p] = part_starts[p];
        skeleton.part_tokens[p] = part_tokens[p];
    }

    resolve_parents(&skeleton, buffer.parent_joints[:buffer.count]);
    return skeleton;
}

<*
 Attach each bone to whichever earlier joint its recorded parent position
 lands on.

 The token stream carries a parent *position*, not an index, so the topology
 has to be recovered by matching. Scanning backwards with strict improvement
 makes the *most recent* joint win a tie, and that detail is load bearing: rigs
 have genuinely coincident joints — a shoulder and an upper arm often sit on
 the same point — and choosing the other one silently reparents a limb.
*>
fn void resolve_parents(Skeleton* skeleton, Vec3[] parent_joints) @local {
    foreach (usz i, parent_joint : parent_joints) {
        if (i == 0) {
            skeleton.parents[0] = -1;
            continue;
        }

        float best = math::FLOAT_MAX;
        usz pick = 0;
        for (usz j = i; j > 0; j--) {
            usz candidate = j - 1;
            float distance = (skeleton.joints[candidate] - parent_joint).length_sq();
            if (distance < best) {
                best = distance;
                pick = candidate;
            }
        }
        skeleton.parents[i] = (int)pick;
    }
}


// --- Generation grammar ---------------------------------------------------
//
// Which tokens may legally follow, given what has been emitted. Without this,
// sampling wanders out of the skeleton vocabulary and into the skin codes —
// the model is trained to emit both, skeleton first, so the boundary is real
// and unmarked in the logits.
//
// The states mirror the reference's `next_posible_token` exactly; a joint is
// three coordinates, and the machine counts them.

enum RigState {
    EXPECT_BOS,
    EXPECT_CLS_OR_PART_OR_JOINT,
    EXPECT_PART_OR_JOINT,
    EXPECT_JOINT,
    EXPECT_JOINT_2,
    EXPECT_JOINT_3,
    EXPECT_BRANCH_OR_PART_OR_JOINT,
}

struct RigGrammar {
    RigState state;
}

fn bool is_part_token(uint token) @local
    => token == TOKEN_SPRING || token == TOKEN_PART_BODY || token == TOKEN_PART_HAND;

fn bool is_class_token(uint token) @local
    => token == TOKEN_CLS_NONE || token == TOKEN_CLS_RIGNET
    || token == TOKEN_CLS_VROID || token == TOKEN_CLS_ARTICULATION;

<*
 Advance the machine over one emitted token.
*>
fn void RigGrammar.accept(&self, uint token) {
    switch (self.state) {
        case EXPECT_BOS:
            self.state = EXPECT_CLS_OR_PART_OR_JOINT;
        case EXPECT_CLS_OR_PART_OR_JOINT:
            if (is_coordinate(token)) {
                self.state = EXPECT_JOINT_2;
            } else if (is_class_token(token)) {
                self.state = EXPECT_PART_OR_JOINT;
            } else {
                self.state = EXPECT_JOINT;
            }
        case EXPECT_PART_OR_JOINT:
            self.state = is_coordinate(token) ? EXPECT_JOINT_2 : EXPECT_PART_OR_JOINT;
        case EXPECT_JOINT:
            self.state = EXPECT_JOINT_2;
        case EXPECT_JOINT_2:
            self.state = EXPECT_JOINT_3;
        case EXPECT_JOINT_3:
            self.state = EXPECT_BRANCH_OR_PART_OR_JOINT;
        case EXPECT_BRANCH_OR_PART_OR_JOINT:
            if (token == TOKEN_BRANCH) {
                self.state = EXPECT_JOINT;
            } else if (is_coordinate(token)) {
                self.state = EXPECT_JOINT_2;
            } else {
                self.state = EXPECT_JOINT;
            }
    }
}

<*
 Mark the tokens that may follow.

 `allowed` is at least RIG_TOTAL_VOCAB long and is fully overwritten, so the
 skin codes and the model's own terminator are excluded by construction —
 nothing outside the skeleton vocabulary is reachable while a skeleton is
 being written.
*>
fn void RigGrammar.allow_next(&self, bool[] allowed) {
    foreach (&slot : allowed) *slot = false;

    switch (self.state) {
        case EXPECT_BOS:
            allowed[TOKEN_BOS] = true;
        case EXPECT_CLS_OR_PART_OR_JOINT:
            allow_classes(allowed);
            allow_parts(allowed);
            allow_coordinates(allowed);
        case EXPECT_PART_OR_JOINT:
            allow_parts(allowed);
            allow_coordinates(allowed);
            allowed[TOKEN_EOS] = true;
        case EXPECT_JOINT:
        case EXPECT_JOINT_2:
        case EXPECT_JOINT_3:
            allow_coordinates(allowed);
        case EXPECT_BRANCH_OR_PART_OR_JOINT:
            allow_coordinates(allowed);
            allow_parts(allowed);
            allowed[TOKEN_BRANCH] = true;
            allowed[TOKEN_EOS] = true;
    }
}

fn void allow_coordinates(bool[] allowed) @local {
    for (uint i = 0; i < NUM_DISCRETE; i++) allowed[i] = true;
}

fn void allow_parts(bool[] allowed) @local {
    allowed[TOKEN_SPRING] = true;
    allowed[TOKEN_PART_BODY] = true;
    allowed[TOKEN_PART_HAND] = true;
}

fn void allow_classes(bool[] allowed) @local {
    allowed[TOKEN_CLS_NONE] = true;
    allowed[TOKEN_CLS_RIGNET] = true;
    allowed[TOKEN_CLS_VROID] = true;
    allowed[TOKEN_CLS_ARTICULATION] = true;
}
