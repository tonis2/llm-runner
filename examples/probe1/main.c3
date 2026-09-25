module probe1;

import std::io;
import vk;
import three;

three::Renderer heap_renderer;

fn int main(String[] args)
{
	io::printfn("stage: arrays");
	uint n = 64 * 256;
	usz quantized_bytes = (usz)(n / 32) * 34;
	char* quantized = (char*)mem::calloc((sz)quantized_bytes);
	defer mem::free(quantized);
	float* weights = (float*)mem::calloc((sz)n);
	defer mem::free(weights);
	float* input = (float*)mem::calloc((sz)256);
	defer mem::free(input);
	float* dequantized = (float*)mem::calloc((sz)n);
	defer mem::free(dequantized);
	float* reference = (float*)mem::calloc((sz)n);
	defer mem::free(reference);
	for (uint k = 0; k < n; k++) weights[k] = (float)(k % 19) * 0.25f;
	for (uint c = 0; c < 256; c++) input[c] = (float)(c % 7) * 0.5f;
	io::printfn("stage: filled");
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
