module probe2;

import std::io;
import vk;
import three;
import libc;

three::Renderer heap_renderer;

fn int main(String[] args)
{
	io::printfn("stage: libc arrays");
	libc::malloc(1);
	uint n = 64 * 256;
	char* quantized = (char*)libc::calloc(1, (usz)(n / 32) * 34);
	defer libc::free(quantized);
	float* weights = (float*)libc::calloc(n, 4);
	defer libc::free(weights);
	io::printfn("stage: filled in libc");
	io::printfn("probe: instance");
	three::Renderer* renderer = &heap_renderer;
	vk::Instance instance = three::create_instance(false, headless: true)!!;
	defer instance.free();

	io::printfn("probe: renderer");
	renderer.init(instance, (vk::SurfaceKHR)null, 512, 512)!!;
	defer renderer.free();
	io::printfn("probe: renderer up");

	renderer.pass.init(&renderer.gpu, &renderer.target)!!;
	defer renderer.pass.free(&renderer.gpu);
	renderer.pass.camera = three::default_camera();

	io::printfn("probe: runtime");
	three::JsRuntime runtime;
	runtime.open(&renderer.pass, renderer)!!;
	defer runtime.close();

	three::RunResult result = runtime.run("return 1 + 1;", "probe.js");
	defer result.free();
	io::printfn("probe: script said %s (ok=%b)", result.value, result.ok);
	return 0;
}
