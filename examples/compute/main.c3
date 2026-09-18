module compute_example;

import std::io;
import std::math;
import std::os;
import libc;
import vk;
import three;

// A real kernel: a Q8_0 weight matrix times an f32 vector, the dequantization
// done inside the shader.
//
// The format is the one the pipelines ship — blocks of 32 weights as an f16
// scale followed by 32 int8 values, 34 bytes each, the same bytes
// `llm::quant_q8_0_block` writes and `dequant_q8_0_block` reads. The kernel
// walks those bytes with a pointer rather than widening every one of them into
// a word, so the load the GPU does is the one the layout asks for.
//
// The check is one of two readings of the same bytes: the host quantizes,
// dequantizes and multiplies the slow way, the script does the same arithmetic
// on the device. Agreement to a float tolerance is what says the shader's
// reading of the layout is the host's.
//
//     c3c build compute --trust=full
//     ./build/compute
//
// The host's own arrays are `libc`'s on purpose. `three::Renderer.init` is
// given the allocator the process is running on and hands pages out of it, so
// host scratch allocated through `mem` and freed once the engine is up is a
// free against the wrong allocator — which is a corruted heap a page later,
// several calls away from the mistake.

const String SLASH_N = "\n";

const uint ROWS = 64;
const uint COLS = 256;
const uint BLOCK = 32;
const uint BLOCK_BYTES = 34;

// One weight per work item. 64x256 is one dispatch of 16384, and no reduction:
// the reference is a scalar loop and so is this, on purpose. The shape whose
// reduction matters is the one with a workgroup per row, and that is a
// different kernel with a different check.
const String KERNEL = `
// Probe: a @storage struct instead of a bare array, same read.
struct In @storage { float data[]; }
In input;
uint weights[];

struct Push @pushconstant
{
    float scale;
}
Push push;

struct ComputeIn { uint3 thread @builtin(global_invocation_id); }

fn uint byte_at(uint at)
{
    return (weights[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
}

fn void main(ComputeIn input2) @compute @threads(64, 1, 1)
{
    uint i = input2.thread.x;
    if (i >= 64u) return;
    float v = input.data[i];
    uint b = byte_at(i);
    _ = v + float(b);
}
`;

// A deterministic weight, small enough that its quantization keeps every digit
// both sides need. `(k % 17) - 8` is exact in float, the fraction is a
// sixteenth, and the pattern is not flat - a kernel that read the wrong
// element would say so.
fn float weight_at(uint k)
{
	return (float)(k % 20u) / 8.0f - 1.0f;
}

// The block the kernel reads: an f16 scale and 32 signed bytes.
fn void quant_q8_0_block(float* values, char* block)
{
	float biggest = 0.0;
	for (uint i = 0; i < BLOCK; i++)
	{
		float magnitude = values[i] < 0 ? -values[i] : values[i];
		if (magnitude > biggest) biggest = magnitude;
	}
	float scale = biggest / 127.0f;
	if (scale < 1e-8f) scale = 1e-8f;

	ushort bits = f32_to_f16_bits(scale);
	block[0] = (char)(bits & 0xFF);
	block[1] = (char)(bits >> 8);
	for (uint i = 0; i < BLOCK; i++)
	{
		float scaled = values[i] / scale;
		int quantized = scaled >= 0 ? (int)(scaled + 0.5f) : (int)(scaled - 0.5f);
		if (quantized > 127) quantized = 127;
		if (quantized < -127) quantized = -127;
		block[2 + i] = (char)quantized;
	}
}

fn void dequant_q8_0_block(char* block, float* output)
{
	uint low = (uint)block[0] & 0xFF;
	uint high = ((uint)block[1] & 0xFF) << 8;
	float scale = f16_bits_to_f32((ushort)(low | high));
	for (uint i = 0; i < BLOCK; i++)
	{
		ichar quantized = (ichar)block[2 + i];
		output[i] = scale * (float)quantized;
	}
}

fn float f16_bits_to_f32(ushort bits)
{
	uint exponent = ((uint)bits >> 10) & 0x1F;
	uint mantissa = (uint)bits & 0x3FF;
	float value;
	if (exponent == 0)
	{
		value = (float)mantissa * 5.9604645e-8f;
	}
	else
	{
		// 2^(exponent - 15), by halves and doublings rather than a pow call:
		// a power of two is exact either way and this is the floor of a test.
		float power = 1.0f;
		uint up = exponent > 15 ? exponent - 15 : 0;
		uint down = exponent < 15 ? 15 - exponent : 0;
		for (uint i = 0; i < up; i++) power *= 2.0f;
		for (uint i = 0; i < down; i++) power *= 0.5f;
		value = (1.0f + (float)mantissa / 1024.0f) * power;
	}
	return (((uint)bits >> 15) & 1) > 0 ? -value : value;
}

fn ushort f32_to_f16_bits(float value)
{
	uint bits = bitcast(value, uint);
	uint sign = (bits >> 31) & 1;
	int exponent = (int)((bits >> 23) & 0xFF) - 127;
	uint mantissa = bits & 0x7FFFFF;
	if (exponent > 15) return (ushort)((sign << 15) | 0x7C00);
	if (exponent < -14) return (ushort)(sign << 15);
	return (ushort)((sign << 15) | ((uint)(exponent + 15) << 10) | (mantissa >> 13));
}

fn String script_source()
{
	// The JavaScript is formatted first, with the kernel left out: the percent
	// marks in it are format specs to tformat and operators to the JavaScript,
	// so the one pass that reads them as specs happens before any operator is
	// in the text, and the splice afterwards is a plain replace.
	String script = `
const n = %d, cols = %d, rows = %d;

const weight = (k) => ((k %% 20) / 8) - 1;

const scratch = new DataView(new ArrayBuffer(4));
function f32Bits(value) {
	scratch.setFloat32(0, value, true);
	return scratch.getUint32(0, true);
}

const packed = new Uint8Array(n / 32 * 34);
for (let block = 0; block < n / 32; block++) {
	let biggest = 0;
	for (let i = 0; i < 32; i++) {
		const magnitude = Math.abs(weight(block * 32 + i));
		if (magnitude > biggest) biggest = magnitude;
	}
	let scale = biggest / 127;
	if (scale < 1e-8) scale = 1e-8;

	const at = block * 34;
	for (let i = 0; i < 32; i++) {
		const scaled = weight(block * 32 + i) / scale;
		let q = scaled >= 0 ? Math.floor(scaled + 0.5) : Math.ceil(scaled - 0.5);
		if (q > 127) q = 127;
		if (q < -127) q = -127;
		packed[at + 2 + i] = q < 0 ? q + 256 : q;
	}
	// The scale as an f16: the host's rounding, as the same bit arithmetic.
	const bits = f32Bits(scale);
	let exponent = ((bits >>> 23) & 0xFF) - 127;
	const mantissa = bits & 0x7FFFFF;
	let half;
	if (exponent > 15) half = 0x7C00;
	else if (exponent < -14) half = 0;
	else half = ((exponent + 15) << 10) | (mantissa >>> 13);
	packed[at] = half & 0xFF;
	packed[at + 1] = (half >> 8) & 0xFF;
}

const weights = three.compute.bytes(packed.length, packed);
const input = three.compute.f32(cols);
for (let c = 0; c < cols; c++) input.bytes[c] = weight(c * 3 + 1);
input.write(input.bytes);
const output = three.compute.f32(rows * cols);

const kernel = three.compute.kernel(SHADER_SLOT, { name: 'q8_matvec' });
kernel.run({ weights, input, output }, {
	threads: rows * cols,
	push: { rows: rows | 0, cols: cols | 0 },
});
output.read();
weights.read();
input.read();

const answers = [];
for (let i = 0; i < rows * cols; i++) answers.push(output.f32(i));
console.log('device weights[0..3]', weights.bytes[0], weights.bytes[1], weights.bytes[2], weights.bytes[3]);
console.log('device input[0..2]', input.f32(0), input.f32(1), input.f32(2));
console.log('device out[0..2]', output.f32(0), output.f32(1), output.f32(2));
return answers;`;
	String formatted = string::tformat(script, ROWS * COLS, COLS, ROWS);
	return formatted.replace(tmem, "SHADER_SLOT", js_string(KERNEL));
}

fn String s2_slice(String text)
{
	sz at = text.index_of("const weight") ?? 0;
	return text[at:60];
}

fn String js_string(String text)
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
	return out.str_view();
}

// A JSON array of numbers, which is all the script returns.
fn bool parse_floats(String text, float[] out)
{
	usz at = 0;
	uint count = 0;
	float? scanned;
	while (at < text.len && count < out.len)
	{
		while (at < text.len && !(text[at] == '-' || (text[at] >= '0' && text[at] <= '9'))) at++;
		usz start = at;
		while (at < text.len && text[at] != ',' && text[at] != ']') at++;
		if (at == start) break;
		scanned = text[start..at - start].to_float();
		if (catch scanned) break;
		out[count++] = scanned;
	}
	return count == out.len;
}

// The renderer the run uses, out of `main`'s frame: the engine it holds is
// pages deep and a stack slot for it is a stack overflow waiting for a build
// that inlines differently.
three::Renderer heap_renderer;

fn int main(String[] args)
{
	uint n = ROWS * COLS;
	usz quantized_bytes = (usz)(n / BLOCK) * (usz)BLOCK_BYTES;

	// The host side, in libc's pages - see the note at the top.
	char* quantized = (char*)libc::calloc(1, quantized_bytes);
	defer libc::free(quantized);
	float* weights = (float*)libc::calloc((usz)n, 4);
	defer libc::free(weights);
	float* input = (float*)libc::calloc((usz)COLS, 4);
	defer libc::free(input);
	float* dequantized = (float*)libc::calloc((usz)n, 4);
	defer libc::free(dequantized);
	float* reference = (float*)libc::calloc((usz)n, 4);
	defer libc::free(reference);

	for (uint k = 0; k < n; k++) weights[k] = weight_at(k);
	for (uint c = 0; c < COLS; c++) input[c] = weight_at(c * 3u + 1u);
	for (uint block = 0; block < n / BLOCK; block++)
	{
		quant_q8_0_block(&weights[block * BLOCK], &quantized[block * BLOCK_BYTES]);
		dequant_q8_0_block(&quantized[block * BLOCK_BYTES], &dequantized[block * BLOCK]);
	}

	for (uint row = 0; row < ROWS; row++)
	{
		for (uint col = 0; col < COLS; col++)
		{
			float sum = 0.0;
			for (uint k = 0; k < COLS; k++)
			{
				sum += dequantized[row * COLS + k] * input[k];
			}
			reference[row * COLS + col] = sum;
		}
	}

	// The device side: a headless engine and a script that reads the same bytes.
	three::Renderer* renderer = &heap_renderer;
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

	three::RunResult result = runtime.run(script_source(), "q8_matvec.js");
	defer result.free();

	if (!result.ok)
	{
		io::printfn("the script failed:\n%s", result.error);
		return 1;
	}

	float* got = (float*)libc::calloc((usz)n, 4);
	defer libc::free(got);
	if (!parse_floats((String)result.value, got[:n]))
	{
		io::printfn("the answer did not parse: %s", (String)result.value);
		return 1;
	}

	float worst = 0.0;
	uint worst_at = 0;
	for (uint i = 0; i < n; i++)
	{
		float diff = math::abs(got[i] - reference[i]);
		float scale = math::max(math::abs(reference[i]), 1.0f);
		if (diff / scale > worst)
		{
			worst = diff / scale;
			worst_at = i;
		}
	}

	io::printfn("%d weights, %d rows of %d: worst relative error %.3g at %d",
		n, ROWS, COLS, worst, worst_at);
	io::printfn("host:   %f", (double)reference[0]);
	io::printfn("device: %f", (double)got[0]);
	io::printfn(worst < 1e-5
		? "the shader reads the format the host writes"
		: "THE SHADER AND THE HOST DISAGREE");
	if (result.log.len > 0) io::printfn("script log:\n%s", result.log);
	return worst < 1e-5 ? 0 : 1;
}
