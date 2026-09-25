module probe_casts;

import three;
import std::io;

// One kernel at a time: the compiler stops at the first error, so a batch tells
// nothing about which line was refused. NAME picks the variant.
const String KERNEL = `
char
float input[];
float output[];

struct Push @pushconstant
{
    uint rows;
    uint cols;
}
Push push;

struct ComputeIn { uint3 thread @builtin(global_invocation_id); }

fn void main(ComputeIn work) @compute @threads(256, 1, 1)
{
    uint index = work.thread.x;
    if (index >= push.rows * push.cols) return;

    uint row = index / push.cols;
    uint col = index % push.cols;
    uint cycles = push.cols / 32u;

    uint base = (row * cycles + col / 32u) * 34u;
    uint low = (uint)weights[base];
    uint high = (uint)weights[base + 1u];
    ushort bits = (ushort)(low | (high << 8));

    float scale;
/*
    half h = (half)bits;
    scale = (float)h;
*/ /*
    scale = unpackHalf2x16((uint)bits).x;
*/ /*
    scale = (float)bits;
*/

    float sum = 0.0;
    for (uint i = 0; i < 32u; i++)
    {
        sum += (float)weights[base + 2u + i] * input[(col / 32u) * 32u + i];
    }

    output[index] = scale * sum;
}
`;

fn int main(String[] args)
{
	three::ShaderModule mod = three::compile_compute_shader("q8_probe", KERNEL);
	defer mod.free();
	if (mod.ok)
	{
		io::printfn("compiled");
		return 0;
	}
	io::printfn("refused:\n%s", mod.diagnostic);
	return 1;
}
