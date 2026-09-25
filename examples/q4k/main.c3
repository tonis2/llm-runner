module q4k_example;

import std::io;
import std::math;
import std::core::mem;
import vk;
import three;

<*
 Q4_K matrix-vector multiply: the format an LLM actually ships, decoded by a
 shady kernel a JavaScript script supplies and dispatches.

 The blocks are generated rather than sampled, and the CPU reference is the
 same arithmetic as `llm::dequant_q4k_block`, so a mismatch points at the
 decode and not at whatever the weights were.
*>
// Q4_K matrix-vector multiply: one thread per row, one 144-byte weight block
// at a time, the 6-bit scales and the nibbles decoded as the CPU does.
//
// The reference is llm::dequant_q4k_block, and the scripts compare against it:
// the answer is a verdict, not a number, because "the wrong nibble" and "the
// wrong scale byte" are the failures this is meant to catch.
const String KERNEL = `

uint weights[];
float input[];
float output[];

struct Push @pushconstant
{
    uint out_dim;
    uint in_dim;
}
Push push;

struct ComputeIn { uint3 thread @builtin(global_invocation_id); }

// One byte of the weight buffer, at a byte offset - the weights are bytes
// however the buffer insists on being typed.
fn uint byte_at(uint at)
{
    return (weights[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
}

// IEEE half to float, by hand: shady has no 16-bit type, and Q4_K stores its
// two scales as halves.
fn float as_float(uint bits)
{
    uint sign = (bits >> 15u) & 0x1u;
    uint exponent = (bits >> 10u) & 0x1Fu;
    uint mantissa = bits & 0x3FFu;
    float s = sign != 0u ? -1.0 : 1.0;

    if (exponent == 0u)
    {
        if (mantissa == 0u) return sign != 0u ? -0.0 : 0.0;
        return s * (float(mantissa) / 1024.0) * (1.0 / 16384.0);
    }
    if (exponent == 31u) return mantissa == 0u ? s * 100000000000.0 : 0.0;
    return s * (1.0 + float(mantissa) / 1024.0) * (1.0 / 16384.0 * 2.0) * float(1u << (exponent - 1u));
}

// A block's 12 packed scale/min bytes, and which of the eight pairs is wanted.
// This is get_scale_min_k4 from llm.slang: the first four scales are a byte
// each, the last four take their high two bits from the byte before.
fn uint scale_at(uint block, uint sub)
{
    return sub < 4u
        ? byte_at(block + 4u + sub) & 63u
        : (byte_at(block + 4u + sub + 4u) & 0xFu) | ((byte_at(block + 4u + sub - 4u) >> 6u) << 4u);
}

fn uint min_at(uint block, uint sub)
{
    return sub < 4u
        ? byte_at(block + 4u + sub + 4u) & 63u
        : (byte_at(block + 4u + sub + 4u) >> 4u) | ((byte_at(block + 4u + sub) >> 6u) << 4u);
}

fn void main(ComputeIn ci) @compute @threads(64, 1, 1)
{
    uint row = ci.thread.x;
    if (row >= push.out_dim) return;

    float sum = 0.0;
    uint blocks_per_row = push.in_dim / 256u;

    for (uint b = 0; b < blocks_per_row; b++)
    {
        uint block = (row * blocks_per_row + b) * 144u;
        float d = as_float(byte_at(block) | (byte_at(block + 1u) << 8u));
        float dmin = as_float(byte_at(block + 2u) | (byte_at(block + 3u) << 8u));

        for (uint j = 0; j < 4u; j++)
        {
            // Four chunks of 64, each spanned by two sub-blocks: the low
            // nibble of every byte is the first 32 values, the high the rest.
            float d1 = d * float(scale_at(block, 2u * j));
            float d2 = d * float(scale_at(block, 2u * j + 1u));
            float dm1 = dmin * float(min_at(block, 2u * j));
            float dm2 = dmin * float(min_at(block, 2u * j + 1u));

            uint qs = block + 16u + j * 32u;
            uint base = b * 256u + j * 64u;

            for (uint l = 0; l < 32u; l++)
            {
                uint packed = byte_at(qs + l);
                sum += (d1 * float(packed & 0xFu) - dm1) * input[base + l];
                sum += (d2 * float(packed >> 4u) - dm2) * input[base + 32u + l];
            }
        }
    }

    output[row] = sum;
}
`;

const uint ROWS = 64;
const uint IN_DIM = 512;
const uint BLOCK_BYTES = 144;
const uint BLOCKS_PER_ROW = IN_DIM / 256;

<*
 Deterministic bytes: a small xorshift, so the weights are the same on every
 run and a failure can be reproduced. The halves are the interesting part -
 a random exponent in a small window, a random mantissa, either sign - and the
 scale bytes are left to cover all 256 combinations of the 6-bit fields.
*>
struct Rng
{
	uint state;
}

fn uint rng_next(Rng* random)
{
	uint x = random.state;
	x ^= x << 13u;
	x ^= x >> 17u;
	x ^= x << 5u;
	random.state = x;
	return x;
}

fn float rng_unit(Rng* random)
{
	return (float)(rng_next(random) >> 8) / 16777216.0f;
}

fn void make_half(Rng* random, char* at)
{
	uint sign = rng_next(random) & 1u;
	uint exponent = 10u + (rng_next(random) % 5u);
	uint mantissa = rng_next(random) & 0x3FFu;
	uint bits = (sign << 15) | (exponent << 10) | mantissa;
	at[0] = (char)(bits & 0xFF);
	at[1] = (char)((bits >> 8) & 0xFF);
}

fn void make_blocks(Rng* random, char[] into)
{
	foreach (&block : (char[BLOCK_BYTES]*)(&into[0]))
	{
		make_half(random, &block[0]);
		make_half(random, &block[2]);
		// The two packed halves of the eight scales and eight mins.
		for (uint i = 0; i < 12; i++) block[4 + i] = (char)(rng_next(random) & 0xFF);
		for (uint i = 0; i < 128; i++) block[16 + i] = (char)(rng_next(random) & 0xFF);
	}
}

// The CPU's answer, accumulated in the same order the kernel accumulates: a
// dequantized split of the block, dotted with the matching input split.
fn void reference(char[] weights, float[] input, float[] out)
{
	for (uint row = 0; row < out.len; row++)
	{
		float sum = 0;
		uint row_bytes = BLOCKS_PER_ROW * BLOCK_BYTES;
		for (uint b = 0; b < BLOCKS_PER_ROW; b++)
		{
			float[256] values;
			dequant_q4k_block(&weights[row * row_bytes + b * BLOCK_BYTES], &values[0]);
			for (uint i = 0; i < 256; i++) sum += values[i] * input[b * 256 + i];
		}
		out[row] = sum;
	}
}

// llm::dequant_q4k_block, kept local: this example is about the JS seam and
// should stay readable without pulling the model library in.
fn void dequant_q4k_block(char* block, float* output)
{
	float d = half_at(block);
	float dmin = half_at(block + 2);
	char* scales = block + 4;
	char* qs = block + 16;

	uint is = 0;
	for (uint j = 0; j < 4; j++)
	{
		uint sc1, m1, sc2, m2;
		uint is0 = is;
		uint is1 = is + 1;

		if (is0 < 4)
		{
			sc1 = (uint)scales[is0] & 63;
			m1 = (uint)scales[is0 + 4] & 63;
		}
		else
		{
			sc1 = ((uint)scales[is0 + 4] & 0xF) | (((uint)scales[is0 - 4] >> 6) << 4);
			m1 = ((uint)scales[is0 + 4] >> 4) | (((uint)scales[is0] >> 6) << 4);
		}
		if (is1 < 4)
		{
			sc2 = (uint)scales[is1] & 63;
			m2 = (uint)scales[is1 + 4] & 63;
		}
		else
		{
			sc2 = ((uint)scales[is1 + 4] & 0xF) | (((uint)scales[is1 - 4] >> 6) << 4);
			m2 = ((uint)scales[is1 + 4] >> 4) | (((uint)scales[is1] >> 6) << 4);
		}

		float d1 = d * (float)sc1;
		float dm1 = dmin * (float)m1;
		float d2 = d * (float)sc2;
		float dm2 = dmin * (float)m2;

		for (uint l = 0; l < 32; l++)
		{
			uint ql = (uint)qs[j * 32 + l];
			output[j * 64 + l] = d1 * (float)(int)(ql & 0xF) - dm1;
		}
		for (uint l = 0; l < 32; l++)
		{
			uint ql = (uint)qs[j * 32 + l];
			output[j * 64 + 32 + l] = d2 * (float)(int)((ql >> 4) & 0xF) - dm2;
		}

		is += 2;
	}
}

fn float half_at(char* at)
{
	uint bits = (uint)(0xFF & (int)at[0]) | (((uint)(0xFF & (int)at[1])) << 8);
	uint sign = (bits >> 15) & 1;
	uint exponent = (bits >> 10) & 0x1F;
	uint mantissa = bits & 0x3FF;
	float s = sign != 0 ? -1.0f : 1.0f;

	if (exponent == 0)
	{
		if (mantissa == 0) return sign != 0 ? -0.0f : 0.0f;
		return s * ((float)mantissa / 1024.0f) * (1.0f / 16384.0f);
	}
	if (exponent == 31) return mantissa == 0 ? s * 100000000000.0f : 0.0f;
	return s * (1.0f + (float)mantissa / 1024.0f) * (1.0f / 16384.0f * 2.0f) * (float)(1u << (exponent - 1));
}

fn String script_source(String kernel_json, String weights_json, String input_json, String expected_json)
{
	return string::tformat(`
const KERNEL = %s;
const WEIGHTS = %s;
const INPUT = %s;

const EXPECTED = %s;
const rows = 64, inDim = 512;
const weights = new Uint8Array(WEIGHTS.length);
for (let i = 0; i < weights.length; i++) weights[i] = WEIGHTS[i];

const w = three.compute.bytes(weights.length, weights);
const x = three.compute.f32(inDim);
const y = three.compute.f32(rows);
for (let i = 0; i < inDim; i++) x.bytes[i] = INPUT[i];
x.write(x.bytes);

const kernel = three.compute.kernel(KERNEL, { name: 'q4k_matvec', pushFields: ['out_dim', 'in_dim'] });
kernel.run({ w, x, y }, { workgroups: rows, push: { out_dim: three.compute.uint(rows), in_dim: three.compute.uint(inDim) } });
y.read();

let worst = 0, worstRow = 0;
for (let i = 0; i < rows; i++) {
	const diff = Math.abs(y.f32(i) - EXPECTED[i]);
	if (diff > worst) { worst = diff; worstRow = i; }
}
return { first: y.f32(0), firstExpected: EXPECTED[0], worst, worstRow };
`, kernel_json, weights_json, input_json, expected_json);
}

<*
 One JSON array of numbers per line's worth of text, one item per value: the
 script parses numbers, and a byte array of 36 KB is cheap to spell out.
*>
fn String json_numbers(float[] values)
{
	DString out;
	out.tinit(values.len * 12 + 2);
	out.append_char('[');
	foreach (v : values)
	{
		if (out.len() > 1) out.append_char(',');
		out.appendf("%.9g", v);
	}
	out.append_char(']');
	return out.copy_str(tmem);
}

fn String json_bytes(char[] values)
{
	DString out;
	out.tinit(values.len * 4 + 2);
	out.append_char('[');
	foreach (v : values)
	{
		if (out.len() > 1) out.append_char(',');
		out.appendf("%d", (int)(0xFF & (int)v));
	}
	out.append_char(']');
	return out.copy_str(tmem);
}

fn String json_string(String text)
{
	DString out;
	out.tinit(text.len + 16);
	out.append_char('"');
	foreach (c : text)
	{
		switch (c)
		{
			case '"':  out.append_string("\\\"");
			case '\\': out.append_string("\\\\");
			case '\n': out.append_string("\\n");
			case '\r': out.append_string("\\r");
			case '\t': out.append_string("\\t");
			default:   out.append_char(c);
		}
	}
	out.append_char('"');
	return out.copy_str(tmem);
}

fn int main(String[] args)
{
	Rng rng = { .state = 0x9E3779B9 };

	char[] weights = mem::new_array(char, ROWS * BLOCKS_PER_ROW * BLOCK_BYTES);
	defer mem::free(weights);
	make_blocks(&rng, weights);

	float[] input = mem::new_array(float, IN_DIM);
	defer mem::free(input);
	foreach (&x : input)
	{
		float value = rng_unit(&rng) * 2.0f - 1.0f;
		*x = value;
	}

	float[] expected = mem::new_array(float, ROWS);
	defer mem::free(expected);
	reference(weights, input, expected);

	three::Renderer* renderer = alloc::new(mem, three::Renderer);
	defer alloc::free(mem, renderer);

	vk::Instance instance = three::create_instance(false, headless: true)!!;
	defer instance.free();
	renderer.init(instance, (vk::SurfaceKHR)null, 512, 512)!!;
	defer renderer.free();

	renderer.pass.init(&renderer.gpu, &renderer.target)!!;
	defer renderer.pass.free(&renderer.gpu);
	renderer.pass.camera = three::default_camera();

	three::JsRuntime runtime;
	runtime.open(&renderer.pass, renderer)!!;
	defer runtime.close();

	String script = script_source(json_string(KERNEL), json_bytes(weights), json_numbers(input), json_numbers(expected));
	three::RunResult result = runtime.run(script, "q4k.js");
	defer result.free();

	if (!result.ok)
	{
		io::printfn("the script failed:\n%s", result.error);
		return 1;
	}

	io::printfn("script said: %s", result.value);

	return 0;
}
