<*
The `__llm` host object: what a pipeline plugin cannot do in JavaScript.

 A plugin owns its graph - which kernels, in what order, over which buffers -
 and the engine's `three.compute` runs it. What is here is the rest: opening a
 model file and streaming its tensors straight onto the GPU, the tokenizer, the
 seeded noise the C3 pipelines drew, and images in and out. All of it is the
 existing C3 code in `lib/`, reached through handles the way `three.compute`
 reaches its buffers.

 ## Why the weights never cross into JavaScript

 A 9 GB model does not fit in a JS heap and has no reason to. `tensorUpload`
 reads the mapped file and writes the GPU buffer in pieces, then hands the
 script a `three.compute` handle to bind. The script sees the tensor's name,
 type and shape; the bytes go from the page cache to VRAM.

 The prelude that makes these verbs readable is `llm/core/llm.js`.

 Part of the `llm` library: the runner executable installs these on its own
 runtime, and `llm::pipeline` installs them on the one a host program runs a
 pipeline on. One set per process - the handle tables are module globals.
*>
module llm::host;

import three;
import llm;
import fastjs::js;
import fastjs::js::opt;
import image;
import image::png;
import image::mp4;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::io::path;
import std::sort;
import std::core::mem;
import std::collections::list;
import std::time::clock;
import std::encoding::base64;

<*
 Which set of host verbs this is. `llm/core/core.json` says which it needs
 (`host`); raise this when a verb is added or changes, so a newer core is
 refused with a reason instead of failing on a missing verb.
*>
const int API = 3;

// --- Handle tables ---------------------------------------------------------------

struct ModelSlot
{
	bool is_gguf;
	bool live;
	mmap::FileMmap mm;           // GGUF only; safetensors keeps its own
	llm::GGUFFile gguf;
	llm::SafetensorsFile st;
}

struct TokenizerSlot
{
	bool live;
	llm::Tokenizer tok;
}

List{ModelSlot} models;
List{TokenizerSlot} tokenizers;
Clock started;

// The video `videoFrame` last read, kept open so frames in order decode on.
mp4::Video video;
String video_path;

<*
 Where `print` lines go instead of stdout, when set: a program running a
 pipeline in-process reads its progress this way.
*>
alias PrintHook = fn void(void* context, String line);
PrintHook print_hook;
void* print_context;

<*
 Put the verbs on `globalThis.__llm`.

 @return? js::THREW : `The engine ran out of memory making the host object - nothing a caller can fix.`
*>
fn void? install(three::JsRuntime* runtime)
{
	started = clock::now();
	models.init(mem, 4);
	tokenizers.init(mem, 2);

	js::Engine* js = runtime.js;
	js::Value host = js.object()!;

	js.@bind(host, "print", &js_print, runtime)!;
	js.@bind(host, "now", &js_now, runtime)!;
	js.@bind(host, "readText", &js_read_text, runtime)!;
	js.@bind(host, "readBytes", &js_read_bytes, runtime)!;
	js.@bind(host, "readRange", &js_read_range, runtime)!;
	js.@bind(host, "fileSize", &js_file_size, runtime)!;
	js.@bind(host, "writeBytes", &js_write_bytes, runtime)!;
	js.@bind(host, "exists", &js_exists, runtime)!;
	js.@bind(host, "listDir", &js_list_dir, runtime)!;
	js.@bind(host, "makeDir", &js_make_dir, runtime)!;
	js.@bind(host, "homeDir", &js_home_dir, runtime)!;
	js.@bind(host, "base64Encode", &js_base64_encode, runtime)!;
	js.@bind(host, "base64Decode", &js_base64_decode, runtime)!;
	js.@bind(host, "evalModule", &js_eval_module, runtime)!;

	js.@bind(host, "modelOpen", &js_model_open, runtime)!;
	js.@bind(host, "modelClose", &js_model_close, runtime)!;
	js.@bind(host, "modelArray", &js_model_array, runtime)!;
	js.@bind(host, "tensorUpload", &js_tensor_upload, runtime)!;
	js.@bind(host, "tensorFloats", &js_tensor_floats, runtime)!;
	js.@bind(host, "embedRows", &js_embed_rows, runtime)!;

	js.@bind(host, "tokenizerOpen", &js_tokenizer_open, runtime)!;
	js.@bind(host, "tokenize", &js_tokenize, runtime)!;
	js.@bind(host, "tokenFind", &js_token_find, runtime)!;
	js.@bind(host, "detokenize", &js_detokenize, runtime)!;

	js.@bind(host, "randomNormal", &js_random_normal, runtime)!;
	js.@bind(host, "imageLoad", &js_image_load, runtime)!;
	js.@bind(host, "imageDecode", &js_image_decode, runtime)!;
	js.@bind(host, "videoInfo", &js_video_info, runtime)!;
	js.@bind(host, "videoFrame", &js_video_frame, runtime)!;
	js.@bind(host, "imageSavePng", &js_image_save_png, runtime)!;
	js.@bind(host, "imageEncodePng", &js_image_encode_png, runtime)!;
	js.@bind(host, "imageResize", &js_image_resize, runtime)!;
	js.@bind(host, "imageToTensor", &js_image_to_tensor, runtime)!;
	js.@bind(host, "tensorToImage", &js_tensor_to_image, runtime)!;

	js.@bind(host, "rigFitEvaluate", &js_rig_fit_evaluate, runtime)!;
	js.@bind(host, "rigFitSteps", &js_rig_fit_steps, runtime)!;

	js.@bind(host, "orthonormalRows", &js_orthonormal_rows, runtime)!;
	js.@bind(host, "imageComposite", &js_image_composite, runtime)!;
	js.@bind(host, "imageCropNearest", &js_image_crop_nearest, runtime)!;
	js.@bind(host, "imageBackdrop", &js_image_backdrop, runtime)!;

	js.set(js.global(), "__llm", host)!;
}

fn void shutdown()
{
	foreach (&slot : models)
	{
		if (!slot.live) continue;
		close_model(slot);
	}
	models.free();
	foreach (&slot : tokenizers)
	{
		if (slot.live) slot.tok.free();
	}
	tokenizers.free();
	close_video();
}

fn void close_video() @local
{
	if (video_path.len == 0) return;
	video.close();
	alloc::free(mem, video_path.ptr);
	video_path = "";
}

fn mp4::Video*? open_video(String path) @local
{
	if (video_path == path) return &video;
	close_video();
	video = mp4::open(path)!;
	video_path = path.copy(mem);
	return &video;
}

fn void close_model(ModelSlot* slot) @local
{
	if (slot.is_gguf)
	{
		slot.gguf.free();
		(void)slot.mm.destroy();
	}
	else
	{
		slot.st.close();
	}
	slot.live = false;
}

// --- Reading arguments -----------------------------------------------------------

<*
 A model handle argument: the index `modelOpen` answered, checked to be open.
*>
struct ModelRef
{
	ModelSlot* slot;
}

<*
 @return? js::THREW
*>
fn void? ModelRef.from_js(&self, js::Call* c, int at)
{
	long index = c.integer(at)!;
	if (index < 0 || index >= models.len() || !models[index].live) return c.fail("this model handle is not open")~;
	self.slot = &models[index];
}

<*
 A tokenizer handle argument, checked to be open.
*>
struct TokenizerRef
{
	llm::Tokenizer* tok;
}

<*
 @return? js::THREW
*>
fn void? TokenizerRef.from_js(&self, js::Call* c, int at)
{
	long index = c.integer(at)!;
	if (index < 0 || index >= tokenizers.len() || !tokenizers[index].live) return c.fail("this tokenizer handle is not open")~;
	self.tok = &tokenizers[index].tok;
}

<*
 An image from a script's `{ width, height, channels, pixels }`. The pixels
 are the script's; the image points at them and must not outlive the call.
*>
struct ImageArg
{
	image::Image img;
}

<*
 @return? js::THREW
*>
fn void? ImageArg.from_js(&self, js::Call* c, int at)
{
	js::Engine* js = c.js;
	js::Value value = c.get(at);
	if (!value.is_object()) return c.fail("%s wants { width, height, channels, pixels }", c.which(at))~;
	uint width = (uint)js.get_number(value, "width")!;
	uint height = (uint)js.get_number(value, "height")!;
	uint channels = (uint)js.get_number(value, "channels")!;
	char[] pixels = js::bytes(js.get(value, "pixels")!);
	image::PixelFormat format;
	switch (channels)
	{
		case 1: format = GRAYSCALE;
		case 2: format = GRAYSCALE_ALPHA;
		case 3: format = RGB;
		case 4: format = RGBA;
		default: return c.fail("%s wants { width, height, channels, pixels } with 1 to 4 channels", c.which(at))~;
	}
	if (width == 0 || height == 0 || pixels.len < (usz)width * height * channels)
	{
		return c.fail("%s wants { width, height, channels, pixels } with width * height * channels pixel bytes", c.which(at))~;
	}
	self.img = { .width = width, .height = height, .bit_depth = 8, .format = format, .pixels = pixels, .allocator = mem };
}

fn js::Value? byte_array(js::Engine* js, char[] data) @local => js.typed_copy(UINT8, data);
fn js::Value? float_array(js::Engine* js, float[] data) @local => js.typed_copy(FLOAT32, ((char*)data.ptr)[:data.len * 4]);
fn js::Value? uint_array(js::Engine* js, uint[] data) @local => js.typed_copy(UINT32, ((char*)data.ptr)[:data.len * 4]);

// --- Plain verbs -----------------------------------------------------------------

<*
 `print(text)` - a line on stdout, now.

 `console.log` collects into the run's log, which a caller reads when the run
 is over; a generation is minutes of work and its progress is worth seeing
 while it happens.
*>
fn void js_print(String line)
{
	if (print_hook)
	{
		print_hook(print_context, line);
		return;
	}
	io::printn(line);
	(void)io::stdout().flush();
}

fn double js_now() => started.to_now().to_sec() * 1000.0;

fn String? js_read_text(js::Engine* js, String path)
{
	char[]? data = file::load(tmem, path);
	if (catch err = data) return js.fail("could not read %s: %s", path, err)~;
	return (String)data;
}

fn js::Value? js_read_bytes(js::Engine* js, String path)
{
	char[]? data = file::load(mem, path);
	if (catch err = data) return js.fail("could not read %s: %s", path, err)~;
	defer mem::free(data);
	return byte_array(js, data);
}

<*
 `readRange(path, offset, length)` - `length` bytes of a file from `offset`, for
 the small parts of a large one: a zip's directory, a checkpoint's pickle.
*>
fn js::Value? js_read_range(js::Engine* js, String path, double offset_number, double length_number)
{
	long offset = (long)offset_number;
	usz length = (usz)length_number;
	File? f = file::open(path, "rb");
	if (catch err = f) return js.fail("could not open %s: %s", path, err)~;
	defer (void)f.close();
	char[] data = mem::new_array(char, (sz)length);
	defer mem::free(data);
	usz got = 0;
	if (catch err = f.seek(offset)) return js.fail("could not seek %s to %d: %s", path, offset, err)~;
	while (got < length)
	{
		sz? n = f.read(data[got..]);
		if (catch err = n) return js.fail("could not read %s: %s", path, err)~;
		if (n == 0) break;
		got += (usz)n;
	}
	return byte_array(js, data[:got]);
}

<*
 `fileSize(path)` - the size of a file in bytes, or -1 when it cannot be read.
*>
fn double js_file_size(String path) => (double)(file::get_size(path) ?? -1);

fn void? js_write_bytes(js::Engine* js, String path, char[] data)
{
	if (catch err = file::save(path, data)) return js.fail("could not write %s: %s", path, err)~;
}

fn bool js_exists(String path) => file::exists(path);

<*
 One entry of `listDir`.
*>
struct DirEntry
{
	String name;
	bool dir;
}

<*
 `listDir(path)` - the entries of a directory as [{ name, dir }], sorted by
 name, without `.` and `..`. A missing directory is an empty list, so a plugin
 or model folder that has not been made yet reads as holding nothing.
*>
fn js::Value? js_list_dir(js::Engine* js, String dir)
{
	js::Value list = js.array()!;
	if (!file::is_dir(dir)) return list;
	path::PathList? entries = path::ls(tmem, (path::Path)dir);
	if (catch entries) return list;
	String[] names = mem::talloc_array(String, entries.len());
	foreach (i, p : entries) names[i] = (String)p;
	sort::quicksort(names, fn int(String a, String b) => a.compare_to(b));
	foreach (name : names)
	{
		if (name == "." || name == "..") continue;
		DirEntry entry = { name, file::is_dir(string::tformat("%s/%s", dir, name)) };
		js.push(list, js.@object_of(entry)!);
	}
	return list;
}

<*
 `makeDir(path)` - make a directory and any missing parents; true when it
 exists afterwards.
*>
fn bool js_make_dir(String dir)
{
	if (!file::is_dir(dir)) (void)path::mkdir(dir, recursive: true);
	return file::is_dir(dir);
}

<*
 `homeDir()` - the user's home directory, or "" when there is none.
*>
fn String js_home_dir()
{
	path::Path? home = path::home_directory(tmem);
	if (catch home) return "";
	return (String)home;
}

fn String js_base64_encode(char[] data) => base64::encode(tmem, data);

fn js::Value? js_base64_decode(js::Engine* js, String text)
{
	// A data URL's header is not base64; what follows the comma is.
	if (try comma = text.index_of(",")) text = text[comma + 1..];
	char[]? data = base64::decode(mem, (char[])text);
	if (catch data) return js.fail("that is not base64")~;
	defer mem::free(data);
	return byte_array(js, data);
}

<*
 `evalModule(source, name)` - evaluate `source` as an ES module named `name`
 and answer the promise for its evaluation. Its imports resolve through the
 runtime's loader like any other module's, and a module already loaded is
 not run again.

 fastjs has no `import()`. `llm.import(specifier)` (`llm/core/llm.js`) stands in
 for it: a one-line module that imports the specifier and hands its namespace
 back through a global.
*>
fn js::Value? js_eval_module(js::Engine* js, String source, String name)
{
	return js.eval(source, name, MODULE);
}

// --- Models ----------------------------------------------------------------------

<*
 `modelOpen(path, table?)` - map a GGUF or safetensors file and describe it.
 With `table`, a safetensors-style JSON header whose offsets count from the
 file's start, any file whose tensors lie in it uncompressed is read that way.

 Answers `{ handle, format, metadata, tensors }`. `tensors` is a list of
 `{ name, type, shape, bytes }`: for GGUF `type` is the GGML type number and
 `shape` is GGUF's own order (fastest dimension first); for safetensors `type`
 is the dtype string and `shape` is PyTorch's order. Metadata arrays are left
 out and described by length, because a tokenizer's vocabulary is a hundred
 thousand strings nobody asked for; `modelArray` reads one on purpose.
*>
fn js::Value? js_model_open(js::Engine* js, String path, Opt {String} table_arg)
{
	ModelSlot slot;
	// A tensor table the caller read itself (a PyTorch checkpoint, see
	// llm/core/weights/torch.js), in safetensors' header form with file offsets.
	String table = table_arg.or("");
	bool gguf = table.len == 0 && path.ends_with(".gguf");
	if (table.len > 0)
	{
		llm::SafetensorsFile? st = llm::table_open(path, table);
		if (catch err = st) return js.fail("could not map %s with the tensor table given: %s", path, err)~;
		slot = { .is_gguf = false, .live = true, .st = st };
	}
	else if (gguf)
	{
		mmap::FileMmap? mm = file::mmap_open(path, "rb");
		if (catch err = mm) return js.fail("could not open %s: %s", path, err)~;
		llm::GGUFFile? parsed = llm::gguf_parse(mm.bytes());
		if (catch err = parsed)
		{
			(void)mm.destroy();
			return js.fail("%s is not a GGUF file this can read: %s", path, err)~;
		}
		slot = { .is_gguf = true, .live = true, .mm = mm, .gguf = parsed };
	}
	else
	{
		llm::SafetensorsFile? st = llm::safetensors_open(path);
		if (catch err = st) return js.fail("could not open %s as safetensors: %s", path, err)~;
		slot = { .is_gguf = false, .live = true, .st = st };
	}

	int index = -1;
	foreach (i, &existing : models)
	{
		if (!existing.live) { index = (int)i; break; }
	}
	if (index < 0)
	{
		models.push(slot);
		index = (int)models.len() - 1;
	}
	else
	{
		models[index] = slot;
	}
	ModelSlot* live = &models[index];

	js::Value result = js.object()!;
	js.set(result, "handle", js::integer(index))!;
	js.set(result, "format", js.string(gguf ? "gguf" : "safetensors")!)!;

	js::Value meta = js.object()!;
	js::Value tensors = js.array()!;
	if (gguf)
	{
		foreach (&kv : live.gguf.metadata)
		{
			String key = kv.key;
			switch (kv.val_type)
			{
				case UINT8: js.set(meta, key, js::number(kv.value.u8_val))!;
				case INT8: js.set(meta, key, js::number(kv.value.i8_val))!;
				case UINT16: js.set(meta, key, js::number(kv.value.u16_val))!;
				case INT16: js.set(meta, key, js::number(kv.value.i16_val))!;
				case UINT32: js.set(meta, key, js::number(kv.value.u32_val))!;
				case INT32: js.set(meta, key, js::number(kv.value.i32_val))!;
				case FLOAT32: js.set(meta, key, js::number(kv.value.f32_val))!;
				case BOOL: js.set(meta, key, js::boolean(kv.value.bool_val))!;
				case STRING: js.set(meta, key, js.string(kv.value.str_val)!)!;
				case UINT64: js.set(meta, key, js::number((double)kv.value.u64_val))!;
				case INT64: js.set(meta, key, js::number((double)kv.value.i64_val))!;
				case FLOAT64: js.set(meta, key, js::number(kv.value.f64_val))!;
				case ARRAY:
					js::Value described = js.object()!;
					js.set(described, "array", js::number((double)kv.array_type.ordinal))!;
					js.set(described, "length", js::number((double)kv.array_len))!;
					js.set(meta, key, described)!;
			}
		}
		foreach (&t : live.gguf.tensors)
		{
			js::Value info = js.object()!;
			js.set(info, "name", js.string(t.name)!)!;
			js.set(info, "type", js::number((double)(uint)t.type))!;
			js::Value shape = js.array(t.n_dims)!;
			for (uint d = 0; d < t.n_dims; d++) js.push(shape, js::number((double)t.shape[d]));
			js.set(info, "shape", shape)!;
			js.set(info, "bytes", js::number((double)t.data_size))!;
			js.push(tensors, info);
		}
	}
	else
	{
		foreach (kv : live.st.metadata)
		{
			js.set(meta, kv.key, js.string(kv.value)!)!;
		}
		for (usz i = 0; i < live.st.n_tensors; i++)
		{
			llm::SafetensorsTensorInfo* t = &live.st.tensors[i];
			js::Value info = js.object()!;
			js.set(info, "name", js.string(t.name)!)!;
			js.set(info, "type", js.string(t.dtype)!)!;
			js::Value shape = js.array(t.n_dims)!;
			for (uint d = 0; d < t.n_dims; d++) js.push(shape, js::number((double)t.shape[d]));
			js.set(info, "shape", shape)!;
			js.set(info, "bytes", js::number((double)(t.data_end - t.data_start)))!;
			js.push(tensors, info);
		}
	}
	js.set(result, "metadata", meta)!;
	js.set(result, "tensors", tensors)!;
	return result;
}

fn void js_model_close(ModelRef model) => close_model(model.slot);

<*
 `modelArray(handle, key)` - one GGUF metadata array, in full.
*>
fn js::Value? js_model_array(js::Engine* js, ModelRef model, String key)
{
	ModelSlot* slot = model.slot;
	if (!slot.is_gguf) return js.fail("modelArray reads GGUF metadata; safetensors has none of that shape")~;
	if (try strings = slot.gguf.read_string_array(key))
	{
		defer mem::free(strings);
		js::Value array = js.array((uint)strings.len)!;
		foreach (s : strings) js.push(array, js.string(s)!);
		return array;
	}
	if (try floats = slot.gguf.read_f32_array(key))
	{
		defer mem::free(floats);
		return float_array(js, floats);
	}
	if (try uints = slot.gguf.read_u32_array(key))
	{
		defer mem::free(uints);
		return uint_array(js, uints);
	}
	return js::UNDEFINED;
}

// --- Tensor bytes, in whatever form was asked for ----------------------------------

<*
 A tensor's bytes on the host, converted as `mode` says.

 `owned` is set when `data` was allocated here and the caller frees it; a raw
 read points straight into the mapped file.

 - `raw`: the file's bytes, whatever the type.
 - `f32`: every element as a float - dequantised for a GGUF quant type,
   widened for F16/BF16/F8.
 - `conv`: `f32`, and a PyTorch conv weight `[out, in, kh, kw]` transposed to
   the `[kh, kw, in, out]` the conv kernels read.
 - `q8`: `f32`, then quantised to Q8_0 blocks.
*>
struct TensorBytes
{
	char* data;
	usz size;
	bool owned;
	usz elements;
}

fn void TensorBytes.free(&self)
{
	if (self.owned && self.data != null) mem::free(self.data);
	*self = {};
}

faultdef NO_SUCH_TENSOR, UNSUPPORTED_TYPE;

fn usz? tensor_elements(ModelSlot* slot, String name) @local
{
	usz n = 1;
	if (slot.is_gguf)
	{
		llm::GGUFTensorInfo* info = slot.gguf.find_tensor(name) ?? NO_SUCH_TENSOR~!;
		for (uint d = 0; d < info.n_dims; d++) n *= (usz)info.shape[d];
	}
	else
	{
		llm::SafetensorsTensorInfo* info = slot.st.find_tensor(name) ?? NO_SUCH_TENSOR~!;
		for (uint d = 0; d < info.n_dims; d++) n *= (usz)info.shape[d];
	}
	return n;
}

fn TensorBytes? tensor_f32(ModelSlot* slot, String name) @local
{
	usz n = tensor_elements(slot, name)!;
	if (slot.is_gguf)
	{
		llm::GGUFTensorInfo* info = slot.gguf.find_tensor(name)!!;
		llm::CpuTensorData cpu = llm::dequant_to_cpu(info, slot.gguf.tensor_data_base)!;
		return { .data = (char*)cpu.data, .size = cpu.size_bytes, .owned = true, .elements = n };
	}
	llm::SafetensorsTensorInfo* info = slot.st.find_tensor(name)!!;
	char* src = slot.st.data_base + (usz)info.data_start;
	float* dst = (float*)mem::calloc((sz)(n * 4));
	switch (info.dtype)
	{
		case "F32": mem::copy(dst, src, (sz)(n * 4));
		case "F16": for (usz j = 0; j < n; j++) dst[j] = llm::cpu_f16_to_f32(src + j * 2);
		case "BF16": for (usz j = 0; j < n; j++) dst[j] = llm::cpu_bf16_to_f32(src + j * 2);
		case "F8_E4M3": for (usz j = 0; j < n; j++) dst[j] = llm::cpu_fp8e4m3_to_f32(src + j);
		default:
			mem::free(dst);
			return UNSUPPORTED_TYPE~;
	}
	return { .data = (char*)dst, .size = n * 4, .owned = true, .elements = n };
}

fn TensorBytes? tensor_bytes(ModelSlot* slot, String name, String mode) @local
{
	switch (mode)
	{
		case "raw":
			if (slot.is_gguf)
			{
				llm::GGUFTensorInfo* info = slot.gguf.find_tensor(name) ?? NO_SUCH_TENSOR~!;
				return { .data = slot.gguf.tensor_data_base + (usz)info.offset, .size = info.data_size,
					.elements = tensor_elements(slot, name)! };
			}
			llm::SafetensorsTensorInfo* info = slot.st.find_tensor(name) ?? NO_SUCH_TENSOR~!;
			return { .data = slot.st.data_base + (usz)info.data_start, .size = (usz)(info.data_end - info.data_start),
				.elements = tensor_elements(slot, name)! };
		case "f32":
			return tensor_f32(slot, name);
		case "conv":
			TensorBytes floats = tensor_f32(slot, name)!;
			usz c_out; usz c_in; usz kh; usz kw;
			if (slot.is_gguf)
			{
				// GGUF keeps a conv weight as [kw, kh, in, out] fastest-first,
				// which is PyTorch's [out, in, kh, kw] in memory.
				llm::GGUFTensorInfo* info = slot.gguf.find_tensor(name)!!;
				if (info.n_dims != 4) return floats;
				kw = (usz)info.shape[0]; kh = (usz)info.shape[1]; c_in = (usz)info.shape[2]; c_out = (usz)info.shape[3];
			}
			else
			{
				llm::SafetensorsTensorInfo* info = slot.st.find_tensor(name)!!;
				if (info.n_dims != 4) return floats;
				c_out = (usz)info.shape[0]; c_in = (usz)info.shape[1]; kh = (usz)info.shape[2]; kw = (usz)info.shape[3];
			}
			float* src = (float*)floats.data;
			float* dst = (float*)mem::calloc((sz)floats.size);
			for (usz oc = 0; oc < c_out; oc++)
			{
				for (usz ic = 0; ic < c_in; ic++)
				{
					for (usz h = 0; h < kh; h++)
					{
						for (usz w = 0; w < kw; w++)
						{
							usz s = oc * (c_in * kh * kw) + ic * (kh * kw) + h * kw + w;
							usz d = h * (kw * c_in * c_out) + w * (c_in * c_out) + ic * c_out + oc;
							dst[d] = src[s];
						}
					}
				}
			}
			usz n = floats.elements;
			floats.free();
			return { .data = (char*)dst, .size = n * 4, .owned = true, .elements = n };
		case "q8":
			TensorBytes floats = tensor_f32(slot, name)!;
			defer floats.free();
			usz n = floats.elements;
			usz blocks = (n + 31) / 32;
			char* q8 = (char*)mem::calloc((sz)(blocks * 34));
			float[32] block;
			for (usz b = 0; b < blocks; b++)
			{
				for (usz i = 0; i < 32; i++)
				{
					usz at = b * 32 + i;
					block[i] = at < n ? ((float*)floats.data)[at] : 0.0f;
				}
				llm::quant_q8_0_block(&block[0], q8 + b * 34);
			}
			return { .data = q8, .size = blocks * 34, .owned = true, .elements = n };
		default:
			return UNSUPPORTED_TYPE~;
	}
}

<*
 What `tensorUpload` answers.
*>
struct UploadedTensor
{
	three::BufferHandle buffer;
	double bytes;
	double elements;
}

<*
 `tensorUpload(handle, name, mode)` - a tensor, onto the GPU, as a buffer handle.

 Answers `{ buffer, bytes, elements }`; `buffer` is what `three.compute.adopt`
 wraps. The write goes through `StorageBuffer.write`, which stages in pieces,
 so a raw weight is read from the mapped file a slice at a time.
*>
fn UploadedTensor? js_tensor_upload(three::JsRuntime* runtime, ModelRef model, String name, Opt {String} mode_arg)
{
	js::Engine* js = runtime.js;
	String mode = mode_arg.or("raw");
	if (mode.len == 0) mode = "raw";

	TensorBytes? bytes = tensor_bytes(model.slot, name, mode);
	if (catch err = bytes)
	{
		if (err == NO_SUCH_TENSOR) return js.fail("the model has no tensor named %s", name)~;
		return js.fail("%s cannot be uploaded as '%s' (%s)", name, mode, err)~;
	}
	defer bytes.free();

	three::Renderer* renderer = runtime.renderer;
	if (renderer == null) return js.fail("tensorUpload needs a GPU device and this runtime has none")~;
	three::StorageBuffer* buffer = alloc::new(mem, three::StorageBuffer);
	if (catch buffer.create(&renderer.gpu, bytes.size))
	{
		alloc::free(mem, buffer);
		return js.fail("could not allocate %d bytes on the GPU for %s", (long)bytes.size, name)~;
	}
	if (catch buffer.write(&renderer.compute, 0, bytes.data, bytes.size))
	{
		buffer.free();
		alloc::free(mem, buffer);
		return js.fail("could not upload %s", name)~;
	}
	return { runtime.adopt_storage_buffer(buffer), (double)bytes.size, (double)bytes.elements };
}

<*
 `tensorFloats(handle, name)` - a small tensor on the host, as floats.

 For the handful of numbers a pipeline reads itself: a batch norm's running
 statistics, a 1x1 convolution applied on the CPU.
*>
fn js::Value? js_tensor_floats(js::Engine* js, ModelRef model, String name)
{
	TensorBytes? floats = tensor_f32(model.slot, name);
	if (catch err = floats) return js.fail("could not read %s as floats (%s)", name, err)~;
	defer floats.free();
	return float_array(js, ((float*)floats.data)[:floats.elements]);
}

<*
 `embedRows(handle, name, ids, buffer)` - look up rows of an embedding table.

 The table stays in the mapped file in its own type; each id's row is
 dequantised on the host and the result, `[ids.length, dim]` floats, is written
 into the buffer behind `buffer` (a `three.compute` buffer handle) from its
 start. `ids` is a Uint32Array's bytes.
*>
fn void? js_embed_rows(three::JsRuntime* runtime, ModelRef model, String name, char[] id_bytes, three::StorageRef target)
{
	js::Engine* js = runtime.js;
	ModelSlot* slot = model.slot;
	if (!slot.is_gguf) return js.fail("embedRows reads a GGUF embedding table")~;
	three::StorageBuffer* buffer = target.slot.buffer;
	llm::GGUFTensorInfo*? info = slot.gguf.find_tensor(name);
	if (catch info) return js.fail("the model has no tensor named %s", name)~;
	uint dim = (uint)info.shape[0];
	usz count = id_bytes.len / 4;
	uint* ids = (uint*)id_bytes.ptr;
	usz total = count * dim;
	if (total * 4 > buffer.size)
	{
		return js.fail("%d rows of %d floats do not fit a buffer of %d bytes", (long)count, dim, (long)buffer.size)~;
	}
	float* rows = (float*)mem::calloc((sz)(total * 4));
	defer mem::free(rows);
	char* table = slot.gguf.tensor_data_base + (usz)info.offset;
	for (usz i = 0; i < count; i++)
	{
		llm::dequant_embedding_row(table, info, ids[i], rows + i * dim, dim);
	}
	if (catch buffer.write(&runtime.renderer.compute, 0, rows, total * 4)) return js.fail("could not upload the embedded rows")~;
}

// --- Tokenizer -------------------------------------------------------------------

<*
 What `tokenizerOpen` answers.
*>
struct OpenedTokenizer
{
	int handle;
	double vocab;
	double bos;
	double eos;
	double pad;
}

fn OpenedTokenizer? js_tokenizer_open(js::Engine* js, ModelRef model)
{
	ModelSlot* slot = model.slot;
	if (!slot.is_gguf) return js.fail("a tokenizer is read out of a GGUF file's metadata")~;
	llm::Tokenizer? tok = llm::load_tokenizer(&slot.gguf);
	if (catch err = tok) return js.fail("this model carries no tokenizer this can read (%s)", err)~;
	tokenizers.push({ .live = true, .tok = tok });
	int index = (int)tokenizers.len() - 1;
	return { index, tok.vocab_size, tok.bos_id, tok.eos_id, tok.pad_id };
}

<*
 `tokenize(tok, text, specials)` - ids, as a Uint32Array. `specials` true reads
 `<|im_start|>`-style markers as the single tokens they are.
*>
fn js::Value? js_tokenize(js::Engine* js, TokenizerRef tokenizer, String text, bool specials)
{
	llm::Tokenizer* tok = tokenizer.tok;
	uint[]? ids = specials ? tok.encode_with_specials(text) : tok.encode(text);
	if (catch err = ids) return js.fail("could not tokenize (%s)", err)~;
	defer mem::free(ids);
	return uint_array(js, ids);
}

fn double js_token_find(TokenizerRef tokenizer, String name) => tokenizer.tok.find_special_token(name);

fn String js_detokenize(TokenizerRef tokenizer, char[] id_bytes)
{
	llm::Tokenizer* tok = tokenizer.tok;
	uint* ids = (uint*)id_bytes.ptr;
	DString text;
	text.tinit(256);
	char[512] piece;
	for (usz i = 0; i < id_bytes.len / 4; i++) text.append_string(tok.decode_token(ids[i], &piece));
	return text.str_view();
}

// --- Noise -----------------------------------------------------------------------

<*
 `randomNormal(count, seed)` - the Gaussian noise the C3 pipelines drew for a
 seed, bit for bit, so a plugin and the pipeline it replaces start from the
 same latent.
*>
fn js::Value? js_random_normal(js::Engine* js, double count_number, double seed_number)
{
	usz count = (usz)count_number;
	uint seed = (uint)(ulong)seed_number;
	float[] noise = mem::new_array(float, (sz)count);
	defer mem::free(noise);
	llm::generate_random_latent(noise, seed);
	return float_array(js, noise);
}

// --- Images ----------------------------------------------------------------------

<*
 Free an image whichever way it was made: the image library's own decoders
 record their allocator, the resizes in `lib/model` do not and allocate on
 `mem`.
*>
fn void drop_image(image::Image* img) @local
{
	if (img.allocator) { img.free(); return; }
	if (img.pixels.len > 0) mem::free(img.pixels);
	img.pixels = {};
}

fn uint channels_of(image::PixelFormat format) @local
{
	switch (format)
	{
		case GRAYSCALE: return 1;
		case GRAYSCALE_ALPHA: return 2;
		case RGB: return 3;
		case RGBA: return 4;
	}
}

fn js::Value? image_value(js::Engine* js, image::Image* img) @local
{
	js::Value result = js.object()!;
	js.set(result, "width", js::number(img.width))!;
	js.set(result, "height", js::number(img.height))!;
	js.set(result, "channels", js::number(channels_of(img.format)))!;
	js.set(result, "pixels", byte_array(js, img.pixels)!)!;
	return result;
}

fn js::Value? js_image_load(js::Engine* js, String path)
{
	image::Image? img = llm::load_image(path);
	if (catch err = img) return js.fail("could not load %s (%s)", path, err)~;
	defer drop_image(&img);
	return image_value(js, &img);
}

<*
 What `videoInfo` answers.
*>
struct VideoInfo
{
	double width;
	double height;
	double frames;
	double fps;
}

// { width, height, frames, fps } of an .mp4/.mov's video.
fn VideoInfo? js_video_info(js::Engine* js, String path)
{
	mp4::Video*? v = open_video(path);
	if (catch err = v) return js.fail("could not read %s: %s", path, err)~;
	return { v.width, v.height, (double)v.frames, v.fps };
}

// Frame k of a video as an image; frames asked in order decode one each.
fn js::Value? js_video_frame(js::Engine* js, String path, int k)
{
	mp4::Video*? v = open_video(path);
	if (catch err = v) return js.fail("could not read %s: %s", path, err)~;
	if (k < 0) return js.fail("%s has no frame %d", path, k)~;
	image::Image? img = v.frame((usz)k);
	if (catch err = img) return js.fail("could not read frame %d of %s: %s", k, path, err)~;
	defer drop_image(&img);
	return image_value(js, &img);
}

fn js::Value? js_image_decode(js::Engine* js, char[] data)
{
	image::Image? img = llm::load_image_from_bytes(data);
	if (catch err = img) return js.fail("could not decode that image (%s)", err)~;
	defer drop_image(&img);
	return image_value(js, &img);
}

fn void? js_image_save_png(js::Engine* js, String path, ImageArg arg)
{
	if (catch err = png::save_file(&arg.img, path)) return js.fail("could not write %s (%s)", path, err)~;
}

fn js::Value? js_image_encode_png(js::Engine* js, ImageArg arg)
{
	char[]? bytes = png::save_bytes(&arg.img);
	if (catch err = bytes) return js.fail("could not encode the image (%s)", err)~;
	defer mem::free(bytes);
	return byte_array(js, bytes);
}

<*
 `imageResize(img, mode, a, b)` - `mode` is `"crop"` (resize the short side to
 cover `a` x `b`, centre-crop to exactly that), `"fit16"` (long side to `a`,
 both sides rounded down to a multiple of 16) or `"exact"` (to `a` x `b`,
 aspect be damned - the image library's own resize).
*>
fn js::Value? js_image_resize(js::Engine* js, ImageArg arg, String mode, double a_number, double b_number)
{
	image::Image* img = &arg.img;
	uint a = (uint)a_number;
	uint b = (uint)b_number;
	image::Image? resized;
	switch (mode)
	{
		case "fit16": resized = llm::resize_aspect_fit_round16(img, a);
		case "exact": resized = img.resize(a, b);
		default: resized = llm::resize_center_crop(img, a, b);
	}
	if (catch err = resized) return js.fail("could not resize (%s)", err)~;
	defer drop_image(&resized);
	return image_value(js, &resized);
}

<*
 `imageToTensor(img, range)` - `[3, h, w]` floats, grey replicated to RGB and
 alpha dropped. `range` `"signed"` (the default) is [-1, 1], what a VAE encoder
 reads; `"unit"` is [0, 1].
*>
fn js::Value? js_image_to_tensor(js::Engine* js, ImageArg arg, String range)
{
	image::Image* img = &arg.img;
	float[]? data;
	if (range == "unit")
	{
		// v / 255 per plane, as `image_to_float_nchw` has it; alpha dropped, grey
		// replicated, so the answer is always three planes.
		float[]? planes = llm::image_to_float_nchw(img);
		if (catch err = planes) return js.fail("could not convert the image (%s)", err)~;
		usz n = (usz)img.width * img.height;
		float[] rgb = mem::new_array(float, (sz)(n * 3));
		for (usz c = 0; c < 3; c++)
		{
			usz src = channels_of(img.format) == 1 ? 0 : c;
			mem::copy(&rgb[c * n], &planes[src * n], (sz)(n * 4));
		}
		mem::free(planes);
		data = rgb;
	}
	else
	{
		data = llm::image_to_float_nchw_signed_rgb(img);
	}
	if (catch err = data) return js.fail("could not convert the image (%s)", err)~;
	defer mem::free(data);
	return float_array(js, data);
}

<*
 `tensorToImage(floats, width, height, channels)` - `[c, h, w]` floats in
 [0, 1] to an 8-bit image, clamped and rounded as the C3 pipelines do. The
 floats are a Float32Array's bytes.
*>
fn js::Value? js_tensor_to_image(js::Engine* js, char[] bytes, double width_number, double height_number, double channels_number)
{
	uint width = (uint)width_number;
	uint height = (uint)height_number;
	uint channels = (uint)channels_number;
	if (bytes.len < (usz)width * height * channels * 4)
	{
		return js.fail("tensorToImage: the floats are fewer than width * height * channels")~;
	}
	float[] data = ((float*)bytes.ptr)[:(usz)width * height * channels];
	image::Image? img = llm::float_tensor_to_image(data, width, height, channels);
	if (catch err = img) return js.fail("could not convert to an image (%s)", err)~;
	defer drop_image(&img);
	return image_value(js, &img);
}
