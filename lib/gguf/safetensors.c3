module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::encoding::json;
import std::collections::object;
import vk;

// Safetensors binary format parser
// Format: 8-byte header_len (LE) | JSON header | raw tensor data
// JSON header maps tensor names to {dtype, shape, data_offsets: [start, end]}

struct SafetensorsTensorInfo {
    String name;
    String dtype;       // "F32", "F16", "BF16"
    ulong[4] shape;
    uint n_dims;
    ulong data_start;   // offset from end of header
    ulong data_end;
}

// Captured key/value pair from the optional __metadata__ object in the
// safetensors header. Diffusers/Kohya/PEFT LoRA files stash things like
// "lora_alpha", "ss_network_alpha", etc. here as stringified scalars.
struct SafetensorsMetaKV {
    String key;
    String value;
}

struct SafetensorsFile {
    SafetensorsTensorInfo[] tensors;
    SafetensorsMetaKV[] metadata; // Empty if header has no __metadata__ section
    char* data_base;     // pointer to start of raw tensor data
    mmap::FileMmap mmap_handle;
    usz n_tensors;
}

// Parsing copies every tensor name and dtype, and every metadata pair, out of
// the JSON object before that object is freed — so each is a separate
// allocation the SafetensorsFile owns. Both cleanups are shared between the
// error paths in safetensors_open and close().
fn void free_tensor_infos(SafetensorsTensorInfo[] tensors, usz count) @local {
    for (usz i = 0; i < count; i++) {
        mem::free(tensors[i].name);
        mem::free(tensors[i].dtype);
    }
    if (tensors.len > 0) mem::free(tensors);
}

fn void free_metadata(SafetensorsMetaKV[] metadata) @local {
    foreach (kv : metadata) {
        mem::free(kv.key);
        mem::free(kv.value);
    }
    if (metadata.len > 0) mem::free(metadata);
}

fn SafetensorsFile? safetensors_open(String path) {
    mmap::FileMmap mm = file::mmap_open(path, "rb")!!;
    // Every failure below returns before the caller has a SafetensorsFile to
    // close, so the mapping has to be released here — otherwise a rejected file
    // stays mapped for the life of the process.
    defer catch (void)mm.destroy();

    char[] data = mm.bytes();

    if (data.len < 8) {
        io::printfn("Error: safetensors file too small");
        return FILE_NOT_FOUND~;
    }

    // Read header length (8 bytes LE)
    ulong header_len = bitcast(*(char[8]*)data[0..], ulong);
    if (header_len > data.len - 8) {
        io::printfn("Error: safetensors header length %d exceeds file size", header_len);
        return FILE_NOT_FOUND~;
    }

    // Parse JSON header using std library parser
    String header_str = (String)data[8 .. 8 + (usz)header_len - 1];
    Object* root = json::parse(mem, header_str)!;
    defer catch root.free();

    if (!root.is_map()) {
        io::printfn("Error: safetensors header is not a JSON object");
        return FILE_NOT_FOUND~;
    }

    // Count tensor entries without materialising a key list: keys() would copy
    // every key just to have them counted and thrown away, and the map already
    // knows its own size.
    bool has_metadata = root.map.has_key("__metadata__");
    usz tensor_count = (usz)root.map.count - (has_metadata ? (usz)1 : (usz)0);

    // Parse optional __metadata__ object — Diffusers/Kohya store LoRA alpha here.
    SafetensorsMetaKV[] metadata = {};
    if (has_metadata) {
        Object* meta_obj = root.get("__metadata__")!;
        if (meta_obj.is_map()) {
            metadata = mem::new_array(SafetensorsMetaKV, (sz)meta_obj.map.count);
            usz mi = 0;
            meta_obj.map.@each(; String mk, Object* mv) {
                // Skip anything that isn't a plain string — the block is
                // conventionally flat, but nothing enforces that.
                if (mv.is_string()) {
                    metadata[mi] = {
                        .key = mk.copy(mem),
                        .value = mv.s.copy(mem),
                    };
                    mi++;
                }
            };
            // Trim if some entries weren't strings.
            if (mi < (usz)meta_obj.map.count) {
                metadata = metadata[0:mi];
            }
        }
    }

    defer catch free_metadata(metadata);

    SafetensorsTensorInfo[] tensors = mem::new_array(SafetensorsTensorInfo, (sz)tensor_count);
    usz n_tensors = 0;

    // The header loop below can fail on any malformed entry, after earlier
    // entries have already had their names copied out of the JSON object.
    defer catch free_tensor_infos(tensors, n_tensors);

    // @each walks the map's own entries, so there is no key array to allocate
    // and no second lookup per tensor — the value comes with the key.
    root.map.@each(; String key, Object* tensor_obj) {
        // Guarded rather than `continue`d: the body is a macro argument, not a
        // loop of its own.
        if (key != "__metadata__") {
            String dtype = tensor_obj.get_string("dtype")!;

            // Parse shape array
            Object* shape_arr = tensor_obj.get("shape")!;
            uint n_dims = (uint)shape_arr.get_len();
            ulong[4] shape = { 0, 0, 0, 0 };
            for (uint i = 0; i < n_dims && i < 4; i++) {
                shape[i] = shape_arr.get_ulong_at(i)!;
            }

            // Parse data_offsets [start, end]
            Object* offsets_arr = tensor_obj.get("data_offsets")!;
            ulong data_start = offsets_arr.get_ulong_at(0)!;
            ulong data_end = offsets_arr.get_ulong_at(1)!;

            // Copy strings — JSON Object owns the originals and will be freed below
            tensors[n_tensors] = {
                .name = key.copy(mem),
                .dtype = dtype.copy(mem),
                .shape = shape,
                .n_dims = n_dims,
                .data_start = data_start,
                .data_end = data_end,
            };
            n_tensors++;
        }
    };

    root.free();

    char* data_base = &data[8 + (usz)header_len];


    return {
        .tensors = tensors,
        .metadata = metadata,
        .data_base = data_base,
        .mmap_handle = mm,
        .n_tensors = n_tensors,
    };
}

fn SafetensorsTensorInfo*? SafetensorsFile.find_tensor(&self, String name) {
    for (usz i = 0; i < self.n_tensors; i++) {
        if (self.tensors[i].name == name) {
            return &self.tensors[i];
        }
    }
    return FILE_NOT_FOUND~;
}

// Look up a string value from the file's __metadata__ section.
// Returns FILE_NOT_FOUND if the key isn't present.
fn String? SafetensorsFile.get_metadata(&self, String key) {
    for (usz i = 0; i < self.metadata.len; i++) {
        if (self.metadata[i].key == key) return self.metadata[i].value;
    }
    return FILE_NOT_FOUND~;
}

fn Tensor? upload_safetensor_f32(DeviceContext* ctx, SafetensorsFile* sf, String name) {
    SafetensorsTensorInfo* info = sf.find_tensor(name)!!;
    char* data_ptr = sf.data_base + (usz)info.data_start;
    usz data_size = (usz)(info.data_end - info.data_start);

    // Calculate number of elements from shape
    usz n_elements = 1;
    for (uint i = 0; i < info.n_dims; i++) {
        n_elements *= (usz)info.shape[i];
    }

    usz f32_size = n_elements * 4;
    float* f32_data;
    bool needs_free = false;

    if (info.dtype == "F32") {
        f32_data = (float*)data_ptr;
    } else if (info.dtype == "F16") {
        f32_data = (float*)mem::calloc((sz)f32_size);
        needs_free = true;
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_f16_to_f32(data_ptr + j * 2);
        }
    } else if (info.dtype == "BF16") {
        f32_data = (float*)mem::calloc((sz)f32_size);
        needs_free = true;
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_bf16_to_f32(data_ptr + j * 2);
        }
    } else if (info.dtype == "F8_E4M3") {
        f32_data = (float*)mem::calloc((sz)f32_size);
        needs_free = true;
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_fp8e4m3_to_f32(data_ptr + j);
        }
    } else {
        io::printfn("Warning: unsupported safetensors dtype '%s' for %s", info.dtype, name);
        return COMPUTE_ERROR~;
    }

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: BUFFER_USAGE_STORAGE_BUFFER_BIT | BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: f32_data,
        data_size: (sz)f32_size,
    )!!;

    if (needs_free) mem::free(f32_data);

    // Reverse shape for GGUF convention (GGUF uses [col, row], safetensors uses [row, col])
    ulong[4] shape = { 0, 0, 0, 0 };
    for (uint i = 0; i < info.n_dims; i++) {
        shape[i] = info.shape[info.n_dims - 1 - i];
    }

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = GGML_F32,
        .n_dims = info.n_dims,
        .shape = shape,
        .size_bytes = f32_size,
    };
}

// Upload conv2d weight from safetensors, transposing from PyTorch [C_out, C_in, kH, kW]
// to GGUF [kH, kW, C_in, C_out] layout expected by the conv2d shader
fn Tensor? upload_safetensor_conv_f32(DeviceContext* ctx, SafetensorsFile* sf, String name) {
    SafetensorsTensorInfo* info = sf.find_tensor(name)!!;
    char* data_ptr = sf.data_base + (usz)info.data_start;

    if (info.n_dims != 4) {
        io::printfn("Warning: conv weight %s has %d dims (expected 4)", name, info.n_dims);
        return upload_safetensor_f32(ctx, sf, name);
    }

    // Safetensors shape: [C_out, C_in, kH, kW]
    usz c_out = (usz)info.shape[0];
    usz c_in  = (usz)info.shape[1];
    usz kh    = (usz)info.shape[2];
    usz kw    = (usz)info.shape[3];
    usz n_elements = c_out * c_in * kh * kw;
    usz f32_size = n_elements * 4;

    // Decode source data to F32
    float* src_data;
    bool needs_free_src = false;

    if (info.dtype == "F32") {
        src_data = (float*)data_ptr;
    } else if (info.dtype == "F16") {
        src_data = (float*)mem::calloc((sz)f32_size);
        needs_free_src = true;
        for (usz j = 0; j < n_elements; j++) {
            src_data[j] = cpu_f16_to_f32(data_ptr + j * 2);
        }
    } else if (info.dtype == "BF16") {
        src_data = (float*)mem::calloc((sz)f32_size);
        needs_free_src = true;
        for (usz j = 0; j < n_elements; j++) {
            src_data[j] = cpu_bf16_to_f32(data_ptr + j * 2);
        }
    } else if (info.dtype == "F8_E4M3") {
        src_data = (float*)mem::calloc((sz)f32_size);
        needs_free_src = true;
        for (usz j = 0; j < n_elements; j++) {
            src_data[j] = cpu_fp8e4m3_to_f32(data_ptr + j);
        }
    } else {
        io::printfn("Warning: unsupported dtype '%s' for conv %s", info.dtype, name);
        return COMPUTE_ERROR~;
    }

    // Transpose: [C_out, C_in, kH, kW] → [kH, kW, C_in, C_out]
    float* dst_data = (float*)mem::calloc((sz)f32_size);
    for (usz oc = 0; oc < c_out; oc++) {
        for (usz ic = 0; ic < c_in; ic++) {
            for (usz h = 0; h < kh; h++) {
                for (usz w = 0; w < kw; w++) {
                    usz src_idx = oc * (c_in * kh * kw) + ic * (kh * kw) + h * kw + w;
                    usz dst_idx = h * (kw * c_in * c_out) + w * (c_in * c_out) + ic * c_out + oc;
                    dst_data[dst_idx] = src_data[src_idx];
                }
            }
        }
    }

    if (needs_free_src) mem::free(src_data);

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: BUFFER_USAGE_STORAGE_BUFFER_BIT | BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: dst_data,
        data_size: (sz)f32_size,
    )!!;

    mem::free(dst_data);

    // Shape in GGUF convention: [kW, kH, C_in, C_out]
    ulong[4] shape = { (ulong)kw, (ulong)kh, (ulong)c_in, (ulong)c_out };

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = GGML_F32,
        .n_dims = 4,
        .shape = shape,
        .size_bytes = f32_size,
    };
}

fn void SafetensorsFile.close(&self) {
    free_tensor_infos(self.tensors, self.n_tensors);
    free_metadata(self.metadata);

    // This is the one that matters at scale: these files run to hundreds of
    // megabytes, and without it every model loaded stays mapped.
    (void)self.mmap_handle.destroy();

    // Leave the struct empty so a second close is harmless rather than a
    // double free — callers often pair this with a defer and an early return.
    *self = {};
}
