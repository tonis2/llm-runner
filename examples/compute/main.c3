module compute_example;

import std::io;
import vk;
import three;

// A headless engine, one JavaScript compute script, and the answer it came
// back with.
//
// This is the seam the two projects meet at: `three` brings the device, the
// shader compiler and the JavaScript runtime up with no window and no frame,
// and the script below writes a kernel in shady and dispatches it. Nothing
// here draws anything - the point is that a pipeline can be a program a person
// reads rather than a C3 function per kernel.
//
//     c3c build compute --trust=full
//     ./build/compute

<*
 The kernel: one buffer read, one written, and a scale from the push block.

 Its binding names - `a` and `result` - are what the script's `run` binds by,
 and its `@threads` is what turns the script's work-item count into groups.
*>
const String KERNEL = `
float a[];
float result[];

struct Push @pushconstant
{
    float scale;
}
Push push;

struct ComputeIn { uint3 thread @builtin(global_invocation_id); }

fn void main(ComputeIn input) @compute @threads(64, 1, 1)
{
    uint i = input.thread.x;
    if (i >= 64u) return;
    result[i] = a[i] * push.scale;
}
`;

<*
 The script, with the kernel spliced in as a quoted string.

 The kernel is multi-line, so it goes in as a JavaScript string literal rather
 than as raw text - which is the same thing a person writing the script by hand
 would have to do.
*>
fn String script_source()
{
	return string::tformat(`
const n = 64;
const a = three.compute.f32(n);
const result = three.compute.f32(n);

for (let i = 0; i < n; i++) a.bytes[i] = i;
a.write(a.bytes);

const kernel = three.compute.kernel(%s, { name: 'scale' });
kernel.run({ a, result }, { threads: n, push: { scale: 2 } });
result.read();

return { first: result.bytes[0], fifth: result.bytes[5], sum: result.bytes[0] + result.bytes[1] };
`, js_literal(KERNEL));
}

fn String js_literal(String text)
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
	// No window and no surface: `headless: true` is the same device a
	// `--headless` three.c3 run builds, offscreen target and all.
	three::Renderer* renderer = alloc::new(mem, three::Renderer);
	defer alloc::free(mem, renderer);

	vk::Instance instance = three::create_instance(false, headless: true)!!;
	defer instance.free();
	renderer.init(instance, (vk::SurfaceKHR)null, 512, 512)!!;
	defer renderer.free();
	// The pass a frame would draw: a runtime reads the rendered scene's stats
	// when it opens, so the scene list has to exist even though nothing here
	// renders. `init` builds the pipeline too, which is the same call a booted
	// engine makes.
	renderer.pass.init(&renderer.gpu, &renderer.target)!!;
	defer renderer.pass.free(&renderer.gpu);
	renderer.pass.camera = three::default_camera();

	three::JsRuntime runtime;
	runtime.open(&renderer.pass, renderer)!!;
	defer runtime.close();

	three::RunResult result = runtime.run(script_source(), "compute.js");
	defer result.free();

	if (!result.ok)
	{
		io::printfn("the script failed:\n%s", result.error);
		return 1;
	}

	io::printfn("script said: %s", result.value);
	if (result.log.len > 0) io::printfn("log:\n%s", result.log);
	return 0;
}
