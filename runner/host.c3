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

 The prelude that makes these verbs readable is `plugins/lib/llm.js`.
*>
module runner;

import three;
import llm;
import qjs;
import image;
import image::png;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::io::path;
import std::sort;
import std::core::mem;
import std::collections::list;
import std::time::clock;
import std::encoding::base64;

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

fn void install(three::JsRuntime* runtime)
{
	started = clock::now();
	models.init(mem, 4);
	tokenizers.init(mem, 2);

	qjs::Value global = runtime.js.global();
	defer runtime.js.release(global);
	qjs::Value host = runtime.js.object();

	runtime.bind(host, "print", &js_print, 1);
	runtime.bind(host, "now", &js_now, 0);
	runtime.bind(host, "readText", &js_read_text, 1);
	runtime.bind(host, "readBytes", &js_read_bytes, 1);
	runtime.bind(host, "writeBytes", &js_write_bytes, 2);
	runtime.bind(host, "exists", &js_exists, 1);
	runtime.bind(host, "listDir", &js_list_dir, 1);
	runtime.bind(host, "makeDir", &js_make_dir, 1);
	runtime.bind(host, "homeDir", &js_home_dir, 0);
	runtime.bind(host, "base64Encode", &js_base64_encode, 1);
	runtime.bind(host, "base64Decode", &js_base64_decode, 1);

	runtime.bind(host, "modelOpen", &js_model_open, 1);
	runtime.bind(host, "modelClose", &js_model_close, 1);
	runtime.bind(host, "modelArray", &js_model_array, 2);
	runtime.bind(host, "tensorUpload", &js_tensor_upload, 3);
	runtime.bind(host, "tensorFloats", &js_tensor_floats, 2);
	runtime.bind(host, "embedRows", &js_embed_rows, 4);

	runtime.bind(host, "tokenizerOpen", &js_tokenizer_open, 1);
	runtime.bind(host, "tokenize", &js_tokenize, 3);
	runtime.bind(host, "tokenFind", &js_token_find, 2);
	runtime.bind(host, "detokenize", &js_detokenize, 2);

	runtime.bind(host, "randomNormal", &js_random_normal, 2);
	runtime.bind(host, "imageLoad", &js_image_load, 1);
	runtime.bind(host, "imageDecode", &js_image_decode, 1);
	runtime.bind(host, "imageSavePng", &js_image_save_png, 2);
	runtime.bind(host, "imageEncodePng", &js_image_encode_png, 1);
	runtime.bind(host, "imageResize", &js_image_resize, 4);
	runtime.bind(host, "imageToTensor", &js_image_to_tensor, 2);
	runtime.bind(host, "tensorToImage", &js_tensor_to_image, 4);

	runtime.js.set_property(global, "__llm", host);
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

fn ModelSlot* model_at(qjs::Context* js, int argc, qjs::Value* args, int at, qjs::Value* out) @local
{
	int index = three::arg_int(js, argc, args, at);
	if (index < 0 || index >= models.len() || !models[index].live)
	{
		*out = js.raise("this model handle is not open");
		return null;
	}
	return &models[index];
}

// --- Reading arguments -----------------------------------------------------------

macro String @text(js, argc, args, at) => three::arg_text(js, argc, args, at, tmem);

fn qjs::Value float_array(qjs::Context* js, float* data, usz count) @local
{
	qjs::Value buffer = js.buffer_copy((char*)data, count * 4);
	defer js.release(buffer);
	return js.typed_array(qjs::ARRAY_FLOAT32, buffer, 0, count);
}

fn qjs::Value byte_array(qjs::Context* js, char* data, usz count) @local
{
	qjs::Value buffer = js.buffer_copy(data, count);
	defer js.release(buffer);
	return js.typed_array(qjs::ARRAY_UINT8, buffer, 0, count);
}

fn qjs::Value uint_array(qjs::Context* js, uint* data, usz count) @local
{
	qjs::Value buffer = js.buffer_copy((char*)data, count * 4);
	defer js.release(buffer);
	return js.typed_array(qjs::ARRAY_UINT32, buffer, 0, count);
}

// --- Plain verbs -----------------------------------------------------------------

<*
 `print(text)` - a line on stdout, now.

 `console.log` collects into the run's log, which a caller reads when the run
 is over; a generation is minutes of work and its progress is worth seeing
 while it happens.
*>
fn void js_print(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	io::printn(@text(&js, argc, args, 0));
	(void)io::stdout().flush();
}

fn void js_now(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	*out = qjs::number_value(started.to_now().to_sec() * 1000.0);
}

fn void js_read_text(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String path = @text(&js, argc, args, 0);
	char[]? data = file::load(tmem, path);
	if (catch err = data)
	{
		*out = js.raise(string::tformat("could not read %s: %s", path, err));
		return;
	}
	*out = js.string((String)data);
}

fn void js_read_bytes(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String path = @text(&js, argc, args, 0);
	char[]? data = file::load(mem, path);
	if (catch err = data)
	{
		*out = js.raise(string::tformat("could not read %s: %s", path, err));
		return;
	}
	defer mem::free(data);
	*out = byte_array(&js, data.ptr, data.len);
}

fn void js_write_bytes(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String path = @text(&js, argc, args, 0);
	char[] data = argc >= 2 ? js.bytes(args[1]) : (char[]){};
	if (catch err = file::save(path, data))
	{
		*out = js.raise(string::tformat("could not write %s: %s", path, err));
	}
}

fn void js_exists(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	*out = qjs::bool_value(file::exists(@text(&js, argc, args, 0)));
}

<*
 `listDir(path)` - the entries of a directory as [{ name, dir }], sorted by
 name, without `.` and `..`. A missing directory is an empty list, so a plugin
 or model folder that has not been made yet reads as holding nothing.
*>
fn void js_list_dir(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String dir = @text(&js, argc, args, 0);
	qjs::Value list = js.array();
	*out = list;
	if (!file::is_dir(dir)) return;
	path::PathList? entries = path::ls(tmem, (path::Path)dir);
	if (catch entries) return;
	String[] names = mem::talloc_array(String, entries.len());
	foreach (i, p : entries) names[i] = (String)p;
	sort::quicksort(names, fn int(String a, String b) => a.compare_to(b));
	uint at = 0;
	foreach (name : names)
	{
		if (name == "." || name == "..") continue;
		qjs::Value entry = js.object();
		js.set_property(entry, "name", js.string(name));
		js.set_property(entry, "dir", qjs::bool_value(file::is_dir(string::tformat("%s/%s", dir, name))));
		js.set_index(list, at++, entry);
	}
}

<*
 `makeDir(path)` - make a directory and any missing parents; true when it
 exists afterwards.
*>
fn void js_make_dir(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String dir = @text(&js, argc, args, 0);
	if (!file::is_dir(dir)) (void)path::mkdir(dir, recursive: true);
	*out = qjs::bool_value(file::is_dir(dir));
}

<*
 `homeDir()` - the user's home directory, or "" when there is none.
*>
fn void js_home_dir(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	path::Path? home = path::home_directory(tmem);
	if (catch home)
	{
		*out = js.string("");
		return;
	}
	*out = js.string((String)home);
}

fn void js_base64_encode(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	char[] data = argc >= 1 ? js.bytes(args[0]) : (char[]){};
	String text = base64::encode(mem, data);
	defer text.free(mem);
	*out = js.string(text);
}

fn void js_base64_decode(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String text = @text(&js, argc, args, 0);
	// A data URL's header is not base64; what follows the comma is.
	if (try comma = text.index_of(",")) text = text[comma + 1..];
	char[]? data = base64::decode(mem, (char[])text);
	if (catch err = data)
	{
		*out = js.raise("that is not base64");
		return;
	}
	defer mem::free(data);
	*out = byte_array(&js, data.ptr, data.len);
}

// --- Models ----------------------------------------------------------------------

<*
 `modelOpen(path)` - map a GGUF or safetensors file and describe it.

 Answers `{ handle, format, metadata, tensors }`. `tensors` is a list of
 `{ name, type, shape, bytes }`: for GGUF `type` is the GGML type number and
 `shape` is GGUF's own order (fastest dimension first); for safetensors `type`
 is the dtype string and `shape` is PyTorch's order. Metadata arrays are left
 out and described by length, because a tokenizer's vocabulary is a hundred
 thousand strings nobody asked for; `modelArray` reads one on purpose.
*>
fn void js_model_open(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String path = @text(&js, argc, args, 0);

	ModelSlot slot;
	bool gguf = path.ends_with(".gguf");
	if (gguf)
	{
		mmap::FileMmap? mm = file::mmap_open(path, "rb");
		if (catch err = mm)
		{
			*out = js.raise(string::tformat("could not open %s: %s", path, err));
			return;
		}
		llm::GGUFFile? parsed = llm::gguf_parse(mm.bytes());
		if (catch err = parsed)
		{
			(void)mm.destroy();
			*out = js.raise(string::tformat("%s is not a GGUF file this can read: %s", path, err));
			return;
		}
		slot = { .is_gguf = true, .live = true, .mm = mm, .gguf = parsed };
	}
	else
	{
		llm::SafetensorsFile? st = llm::safetensors_open(path);
		if (catch err = st)
		{
			*out = js.raise(string::tformat("could not open %s as safetensors: %s", path, err));
			return;
		}
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

	qjs::Value result = js.object();
	js.set_property(result, "handle", qjs::int_value(index));
	js.set_property(result, "format", js.string(gguf ? "gguf" : "safetensors"));

	qjs::Value meta = js.object();
	qjs::Value tensors = js.array();
	if (gguf)
	{
		foreach (&kv : live.gguf.metadata)
		{
			ZString key = kv.key.zstr_tcopy();
			switch (kv.val_type)
			{
				case UINT8: js.set_property(meta, key, qjs::number_value(kv.value.u8_val));
				case INT8: js.set_property(meta, key, qjs::number_value(kv.value.i8_val));
				case UINT16: js.set_property(meta, key, qjs::number_value(kv.value.u16_val));
				case INT16: js.set_property(meta, key, qjs::number_value(kv.value.i16_val));
				case UINT32: js.set_property(meta, key, qjs::number_value(kv.value.u32_val));
				case INT32: js.set_property(meta, key, qjs::number_value(kv.value.i32_val));
				case FLOAT32: js.set_property(meta, key, qjs::number_value(kv.value.f32_val));
				case BOOL: js.set_property(meta, key, qjs::bool_value(kv.value.bool_val));
				case STRING: js.set_property(meta, key, js.string(kv.value.str_val));
				case UINT64: js.set_property(meta, key, qjs::number_value((double)kv.value.u64_val));
				case INT64: js.set_property(meta, key, qjs::number_value((double)kv.value.i64_val));
				case FLOAT64: js.set_property(meta, key, qjs::number_value(kv.value.f64_val));
				case ARRAY:
					qjs::Value described = js.object();
					js.set_property(described, "array", qjs::number_value((double)kv.array_type.ordinal));
					js.set_property(described, "length", qjs::number_value((double)kv.array_len));
					js.set_property(meta, key, described);
			}
		}
		foreach (i, &t : live.gguf.tensors)
		{
			qjs::Value info = js.object();
			js.set_property(info, "name", js.string(t.name));
			js.set_property(info, "type", qjs::number_value((double)(uint)t.type));
			qjs::Value shape = js.array();
			for (uint d = 0; d < t.n_dims; d++) js.set_index(shape, d, qjs::number_value((double)t.shape[d]));
			js.set_property(info, "shape", shape);
			js.set_property(info, "bytes", qjs::number_value((double)t.data_size));
			js.set_index(tensors, (uint)i, info);
		}
	}
	else
	{
		foreach (kv : live.st.metadata)
		{
			js.set_property(meta, kv.key.zstr_tcopy(), js.string(kv.value));
		}
		for (usz i = 0; i < live.st.n_tensors; i++)
		{
			llm::SafetensorsTensorInfo* t = &live.st.tensors[i];
			qjs::Value info = js.object();
			js.set_property(info, "name", js.string(t.name));
			js.set_property(info, "type", js.string(t.dtype));
			qjs::Value shape = js.array();
			for (uint d = 0; d < t.n_dims; d++) js.set_index(shape, d, qjs::number_value((double)t.shape[d]));
			js.set_property(info, "shape", shape);
			js.set_property(info, "bytes", qjs::number_value((double)(t.data_end - t.data_start)));
			js.set_index(tensors, (uint)i, info);
		}
	}
	js.set_property(result, "metadata", meta);
	js.set_property(result, "tensors", tensors);
	*out = result;
}

fn void js_model_close(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	ModelSlot* slot = model_at(&js, argc, args, 0, out);
	if (slot == null) return;
	close_model(slot);
}

<*
 `modelArray(handle, key)` - one GGUF metadata array, in full.
*>
fn void js_model_array(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	ModelSlot* slot = model_at(&js, argc, args, 0, out);
	if (slot == null) return;
	if (!slot.is_gguf)
	{
		*out = js.raise("modelArray reads GGUF metadata; safetensors has none of that shape");
		return;
	}
	String key = @text(&js, argc, args, 1);
	if (try strings = slot.gguf.read_string_array(key))
	{
		defer mem::free(strings);
		qjs::Value array = js.array();
		foreach (i, s : strings) js.set_index(array, (uint)i, js.string(s));
		*out = array;
		return;
	}
	if (try floats = slot.gguf.read_f32_array(key))
	{
		defer mem::free(floats);
		*out = float_array(&js, floats.ptr, floats.len);
		return;
	}
	if (try uints = slot.gguf.read_u32_array(key))
	{
		defer mem::free(uints);
		*out = uint_array(&js, uints.ptr, uints.len);
		return;
	}
	*out = qjs::undefined_value();
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
 `tensorUpload(handle, name, mode)` - a tensor, onto the GPU, as a buffer handle.

 Answers `{ buffer, bytes, elements }`; `buffer` is what `three.compute.adopt`
 wraps. The write goes through `StorageBuffer.write`, which stages in pieces,
 so a raw weight is read from the mapped file a slice at a time.
*>
fn void js_tensor_upload(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	three::JsRuntime* runtime = (three::JsRuntime*)opaque;
	qjs::Context js = qjs::wrap(ctx);
	ModelSlot* slot = model_at(&js, argc, args, 0, out);
	if (slot == null) return;
	String name = @text(&js, argc, args, 1);
	String mode = argc >= 3 ? @text(&js, argc, args, 2) : "raw";
	if (mode.len == 0) mode = "raw";

	TensorBytes? bytes = tensor_bytes(slot, name, mode);
	if (catch err = bytes)
	{
		*out = js.raise(err == NO_SUCH_TENSOR
			? string::tformat("the model has no tensor named %s", name)
			: string::tformat("%s cannot be uploaded as '%s' (%s)", name, mode, err));
		return;
	}
	defer bytes.free();

	three::Renderer* renderer = runtime.renderer;
	three::StorageBuffer* buffer = alloc::new(mem, three::StorageBuffer);
	if (catch buffer.create(&renderer.gpu, bytes.size))
	{
		alloc::free(mem, buffer);
		*out = js.raise(string::tformat("could not allocate %d bytes on the GPU for %s", (long)bytes.size, name));
		return;
	}
	if (catch buffer.write(&renderer.compute, 0, bytes.data, bytes.size))
	{
		buffer.free();
		alloc::free(mem, buffer);
		*out = js.raise(string::tformat("could not upload %s", name));
		return;
	}

	qjs::Value result = js.object();
	js.set_property(result, "buffer", qjs::int_value(runtime.adopt_storage_buffer(buffer)));
	js.set_property(result, "bytes", qjs::number_value((double)bytes.size));
	js.set_property(result, "elements", qjs::number_value((double)bytes.elements));
	*out = result;
}

<*
 `tensorFloats(handle, name)` - a small tensor on the host, as floats.

 For the handful of numbers a pipeline reads itself: a batch norm's running
 statistics, a 1x1 convolution applied on the CPU.
*>
fn void js_tensor_floats(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	ModelSlot* slot = model_at(&js, argc, args, 0, out);
	if (slot == null) return;
	String name = @text(&js, argc, args, 1);
	TensorBytes? floats = tensor_f32(slot, name);
	if (catch err = floats)
	{
		*out = js.raise(string::tformat("could not read %s as floats (%s)", name, err));
		return;
	}
	defer floats.free();
	*out = float_array(&js, (float*)floats.data, floats.elements);
}

<*
 `embedRows(handle, name, ids, buffer)` - look up rows of an embedding table.

 The table stays in the mapped file in its own type; each id's row is
 dequantised on the host and the result, `[ids.length, dim]` floats, is written
 into the buffer behind `buffer` from its start. `ids` is a Uint32Array's bytes.
*>
fn void js_embed_rows(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	three::JsRuntime* runtime = (three::JsRuntime*)opaque;
	qjs::Context js = qjs::wrap(ctx);
	ModelSlot* slot = model_at(&js, argc, args, 0, out);
	if (slot == null) return;
	if (!slot.is_gguf)
	{
		*out = js.raise("embedRows reads a GGUF embedding table");
		return;
	}
	String name = @text(&js, argc, args, 1);
	char[] id_bytes = argc >= 3 ? js.bytes(args[2]) : (char[]){};
	three::StorageBuffer* buffer = runtime.storage_buffer_of(three::arg_int(&js, argc, args, 3));
	if (buffer == null)
	{
		*out = js.raise("embedRows wants a live buffer to write into");
		return;
	}
	llm::GGUFTensorInfo*? info = slot.gguf.find_tensor(name);
	if (catch info)
	{
		*out = js.raise(string::tformat("the model has no tensor named %s", name));
		return;
	}
	uint dim = (uint)info.shape[0];
	usz count = id_bytes.len / 4;
	uint* ids = (uint*)id_bytes.ptr;
	usz total = count * dim;
	if (total * 4 > buffer.size)
	{
		*out = js.raise(string::tformat("%d rows of %d floats do not fit a buffer of %d bytes", (long)count, dim, (long)buffer.size));
		return;
	}
	float* rows = (float*)mem::calloc((sz)(total * 4));
	defer mem::free(rows);
	char* table = slot.gguf.tensor_data_base + (usz)info.offset;
	for (usz i = 0; i < count; i++)
	{
		llm::dequant_embedding_row(table, info, ids[i], rows + i * dim, dim);
	}
	if (catch buffer.write(&runtime.renderer.compute, 0, rows, total * 4))
	{
		*out = js.raise("could not upload the embedded rows");
	}
}

// --- Tokenizer -------------------------------------------------------------------

fn void js_tokenizer_open(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	ModelSlot* slot = model_at(&js, argc, args, 0, out);
	if (slot == null) return;
	if (!slot.is_gguf)
	{
		*out = js.raise("a tokenizer is read out of a GGUF file's metadata");
		return;
	}
	llm::Tokenizer? tok = llm::load_tokenizer(&slot.gguf);
	if (catch err = tok)
	{
		*out = js.raise(string::tformat("this model carries no tokenizer this can read (%s)", err));
		return;
	}
	tokenizers.push({ .live = true, .tok = tok });
	uint index = (uint)tokenizers.len() - 1;

	qjs::Value result = js.object();
	js.set_property(result, "handle", qjs::int_value((int)index));
	js.set_property(result, "vocab", qjs::number_value(tok.vocab_size));
	js.set_property(result, "bos", qjs::number_value(tok.bos_id));
	js.set_property(result, "eos", qjs::number_value(tok.eos_id));
	js.set_property(result, "pad", qjs::number_value(tok.pad_id));
	*out = result;
}

fn llm::Tokenizer* tokenizer_at(qjs::Context* js, int argc, qjs::Value* args, qjs::Value* out) @local
{
	int index = three::arg_int(js, argc, args, 0);
	if (index < 0 || index >= tokenizers.len() || !tokenizers[index].live)
	{
		*out = js.raise("this tokenizer handle is not open");
		return null;
	}
	return &tokenizers[index].tok;
}

<*
 `tokenize(tok, text, specials)` - ids, as a Uint32Array. `specials` true reads
 `<|im_start|>`-style markers as the single tokens they are.
*>
fn void js_tokenize(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	llm::Tokenizer* tok = tokenizer_at(&js, argc, args, out);
	if (tok == null) return;
	String text = @text(&js, argc, args, 1);
	bool specials = argc >= 3 && js.truth(args[2]);
	uint[]? ids = specials ? tok.encode_with_specials(text) : tok.encode(text);
	if (catch err = ids)
	{
		*out = js.raise(string::tformat("could not tokenize (%s)", err));
		return;
	}
	defer mem::free(ids);
	*out = uint_array(&js, ids.ptr, ids.len);
}

fn void js_token_find(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	llm::Tokenizer* tok = tokenizer_at(&js, argc, args, out);
	if (tok == null) return;
	*out = qjs::number_value(tok.find_special_token(@text(&js, argc, args, 1)));
}

fn void js_detokenize(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	llm::Tokenizer* tok = tokenizer_at(&js, argc, args, out);
	if (tok == null) return;
	char[] id_bytes = argc >= 2 ? js.bytes(args[1]) : (char[]){};
	uint* ids = (uint*)id_bytes.ptr;
	DString text;
	text.tinit(256);
	char[512] piece;
	for (usz i = 0; i < id_bytes.len / 4; i++) text.append_string(tok.decode_token(ids[i], &piece));
	*out = js.string(text.str_view());
}

// --- Noise -----------------------------------------------------------------------

<*
 `randomNormal(count, seed)` - the Gaussian noise the C3 pipelines drew for a
 seed, bit for bit, so a plugin and the pipeline it replaces start from the
 same latent.
*>
fn void js_random_normal(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	usz count = (usz)three::arg_number(&js, argc, args, 0);
	uint seed = (uint)(ulong)three::arg_number(&js, argc, args, 1);
	float[] noise = mem::new_array(float, (sz)count);
	defer mem::free(noise);
	llm::generate_random_latent(noise, seed);
	*out = float_array(&js, noise.ptr, count);
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

fn qjs::Value image_value(qjs::Context* js, image::Image* img) @local
{
	qjs::Value result = js.object();
	js.set_property(result, "width", qjs::number_value(img.width));
	js.set_property(result, "height", qjs::number_value(img.height));
	js.set_property(result, "channels", qjs::number_value(channels_of(img.format)));
	js.set_property(result, "pixels", byte_array(js, img.pixels.ptr, img.pixels.len));
	return result;
}

<*
 An image from a script's `{ width, height, channels, pixels }`. The pixels
 are the script's; the image points at them and must not outlive the call.
*>
fn image::Image? image_of(qjs::Context* js, qjs::Value value) @local
{
	qjs::Value w = js.property(value, "width");
	qjs::Value h = js.property(value, "height");
	qjs::Value c = js.property(value, "channels");
	qjs::Value p = js.property(value, "pixels");
	defer { js.release(w); js.release(h); js.release(c); js.release(p); }
	uint width = (uint)js.number(w);
	uint height = (uint)js.number(h);
	uint channels = (uint)js.number(c);
	char[] pixels = js.bytes(p);
	if (width == 0 || height == 0 || pixels.len < (usz)width * height * channels) return UNSUPPORTED_TYPE~;
	image::PixelFormat format;
	switch (channels)
	{
		case 1: format = GRAYSCALE;
		case 2: format = GRAYSCALE_ALPHA;
		case 3: format = RGB;
		case 4: format = RGBA;
		default: return UNSUPPORTED_TYPE~;
	}
	return { .width = width, .height = height, .bit_depth = 8, .format = format, .pixels = pixels, .allocator = mem };
}

fn void js_image_load(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String path = @text(&js, argc, args, 0);
	image::Image? img = llm::load_image(path);
	if (catch err = img)
	{
		*out = js.raise(string::tformat("could not load %s (%s)", path, err));
		return;
	}
	defer drop_image(&img);
	*out = image_value(&js, &img);
}

fn void js_image_decode(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	char[] data = argc >= 1 ? js.bytes(args[0]) : (char[]){};
	image::Image? img = llm::load_image_from_bytes(data);
	if (catch err = img)
	{
		*out = js.raise(string::tformat("could not decode that image (%s)", err));
		return;
	}
	defer drop_image(&img);
	*out = image_value(&js, &img);
}

fn void js_image_save_png(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	String path = @text(&js, argc, args, 0);
	image::Image? img = argc >= 2 ? image_of(&js, args[1]) : UNSUPPORTED_TYPE~;
	if (catch img)
	{
		*out = js.raise("imageSavePng wants { width, height, channels, pixels }");
		return;
	}
	if (catch err = png::save_file(&img, path))
	{
		*out = js.raise(string::tformat("could not write %s (%s)", path, err));
	}
}

fn void js_image_encode_png(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	image::Image? img = argc >= 1 ? image_of(&js, args[0]) : UNSUPPORTED_TYPE~;
	if (catch img)
	{
		*out = js.raise("imageEncodePng wants { width, height, channels, pixels }");
		return;
	}
	char[]? bytes = png::save_bytes(&img);
	if (catch err = bytes)
	{
		*out = js.raise(string::tformat("could not encode the image (%s)", err));
		return;
	}
	defer mem::free(bytes);
	*out = byte_array(&js, bytes.ptr, bytes.len);
}

<*
 `imageResize(img, mode, a, b)` - `mode` is `"crop"` (resize the short side to
 cover `a` x `b`, centre-crop to exactly that), `"fit16"` (long side to `a`,
 both sides rounded down to a multiple of 16) or `"exact"` (to `a` x `b`,
 aspect be damned - the image library's own resize).
*>
fn void js_image_resize(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	image::Image? img = argc >= 1 ? image_of(&js, args[0]) : UNSUPPORTED_TYPE~;
	if (catch img)
	{
		*out = js.raise("imageResize wants { width, height, channels, pixels }");
		return;
	}
	String mode = @text(&js, argc, args, 1);
	uint a = (uint)three::arg_number(&js, argc, args, 2);
	uint b = (uint)three::arg_number(&js, argc, args, 3);
	image::Image? resized;
	switch (mode)
	{
		case "fit16": resized = llm::resize_aspect_fit_round16(&img, a);
		case "exact": resized = img.resize(a, b);
		default: resized = llm::resize_center_crop(&img, a, b);
	}
	if (catch err = resized)
	{
		*out = js.raise(string::tformat("could not resize (%s)", err));
		return;
	}
	defer drop_image(&resized);
	*out = image_value(&js, &resized);
}

<*
 `imageToTensor(img, range)` - `[3, h, w]` floats, grey replicated to RGB and
 alpha dropped. `range` `"signed"` (the default) is [-1, 1], what a VAE encoder
 reads; `"unit"` is [0, 1].
*>
fn void js_image_to_tensor(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	image::Image? img = argc >= 1 ? image_of(&js, args[0]) : UNSUPPORTED_TYPE~;
	if (catch img)
	{
		*out = js.raise("imageToTensor wants { width, height, channels, pixels }");
		return;
	}
	bool unit = argc >= 2 && @text(&js, argc, args, 1) == "unit";
	float[]? data;
	if (unit)
	{
		// v / 255 per plane, as `image_to_float_nchw` has it; alpha dropped, grey
		// replicated, so the answer is always three planes.
		float[]? planes = llm::image_to_float_nchw(&img);
		if (catch err = planes)
		{
			*out = js.raise(string::tformat("could not convert the image (%s)", err));
			return;
		}
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
		data = llm::image_to_float_nchw_signed_rgb(&img);
	}
	if (catch err = data)
	{
		*out = js.raise(string::tformat("could not convert the image (%s)", err));
		return;
	}
	defer mem::free(data);
	*out = float_array(&js, data.ptr, data.len);
}

<*
 `tensorToImage(floats, width, height, channels)` - `[c, h, w]` floats in
 [0, 1] to an 8-bit image, clamped and rounded as the C3 pipelines do. The
 floats are a Float32Array's bytes.
*>
fn void js_tensor_to_image(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	char[] bytes = argc >= 1 ? js.bytes(args[0]) : (char[]){};
	uint width = (uint)three::arg_number(&js, argc, args, 1);
	uint height = (uint)three::arg_number(&js, argc, args, 2);
	uint channels = (uint)three::arg_number(&js, argc, args, 3);
	if (bytes.len < (usz)width * height * channels * 4)
	{
		*out = js.raise("tensorToImage: the floats are fewer than width * height * channels");
		return;
	}
	float[] data = ((float*)bytes.ptr)[:(usz)width * height * channels];
	image::Image? img = llm::float_tensor_to_image(data, width, height, channels);
	if (catch err = img)
	{
		*out = js.raise(string::tformat("could not convert to an image (%s)", err));
		return;
	}
	defer drop_image(&img);
	*out = image_value(&js, &img);
}
