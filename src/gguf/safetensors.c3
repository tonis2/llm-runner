module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
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

struct SafetensorsFile {
    SafetensorsTensorInfo[] tensors;
    char* data_base;     // pointer to start of raw tensor data
    mmap::FileMmap mmap_handle;
    usz n_tensors;
}

fn SafetensorsFile? safetensors_open(String path) {
    mmap::FileMmap mm = file::mmap_open(path, "rb")!!;
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

    // Parse JSON header manually for tensor info
    String header_str = (String)data[8 .. 8 + (usz)header_len - 1];
    SafetensorsTensorInfo[] tensors = mem::new_array(SafetensorsTensorInfo, 256);
    usz n_tensors = 0;
    // The safetensors JSON is: {"tensor_name": {"dtype":"F32","shape":[a,b],"data_offsets":[start,end]}, ...}
    n_tensors = 0;
    usz pos = 0;
    while (pos < header_str.len) {
        // Find next key (quoted string at top level)
        if (header_str[pos] != '"') { pos++; continue; }

        // Extract key
        usz key_start = pos + 1;
        pos++;
        while (pos < header_str.len && header_str[pos] != '"') pos++;
        String key = header_str[key_start .. pos - 1];
        pos++; // skip closing quote

        // Skip "__metadata__" or any non-tensor key
        if (key == "__metadata__") {
            // Skip colon, then skip the value object (handle nested braces and quoted strings)
            while (pos < header_str.len && header_str[pos] != '{') pos++;
            if (pos < header_str.len) {
                uint depth = 1; pos++;
                while (pos < header_str.len && depth > 0) {
                    if (header_str[pos] == '"') {
                        pos++;
                        while (pos < header_str.len && header_str[pos] != '"') {
                            if (header_str[pos] == '\\') pos++; // skip escaped chars
                            pos++;
                        }
                    } else if (header_str[pos] == '{') {
                        depth++;
                    } else if (header_str[pos] == '}') {
                        depth--;
                    }
                    pos++;
                }
            }
            continue;
        }

        // Find the value object (skip to opening brace)
        while (pos < header_str.len && header_str[pos] != '{') pos++;
        if (pos >= header_str.len) break;
        pos++; // skip '{'

        // Parse dtype, shape, data_offsets from this object
        String dtype = "";
        ulong[4] shape = { 0, 0, 0, 0 };
        uint n_dims = 0;
        ulong data_start = 0;
        ulong data_end = 0;

        // Simple state machine parser for the tensor info object
        uint depth = 1;
        while (pos < header_str.len && depth > 0) {
            // Skip whitespace
            while (pos < header_str.len && (header_str[pos] == ' ' || header_str[pos] == '\n' || header_str[pos] == '\r' || header_str[pos] == '\t' || header_str[pos] == ',')) pos++;

            if (pos >= header_str.len) break;
            if (header_str[pos] == '}') { depth--; pos++; continue; }

            // Parse key
            if (header_str[pos] != '"') { pos++; continue; }
            usz fk_start = pos + 1;
            pos++;
            while (pos < header_str.len && header_str[pos] != '"') pos++;
            String field_key = header_str[fk_start .. pos - 1];
            pos++; // skip closing quote

            // Skip colon
            while (pos < header_str.len && header_str[pos] != ':') pos++;
            pos++;
            while (pos < header_str.len && header_str[pos] == ' ') pos++;

            if (field_key == "dtype") {
                // Parse string value
                if (header_str[pos] == '"') {
                    usz vs = pos + 1;
                    pos++;
                    while (pos < header_str.len && header_str[pos] != '"') pos++;
                    dtype = header_str[vs .. pos - 1];
                    pos++;
                }
            } else if (field_key == "shape") {
                // Parse array of ints
                if (header_str[pos] == '[') {
                    pos++;
                    n_dims = 0;
                    while (pos < header_str.len && header_str[pos] != ']') {
                        while (pos < header_str.len && (header_str[pos] == ' ' || header_str[pos] == ',')) pos++;
                        if (header_str[pos] == ']') break;
                        ulong val = 0;
                        while (pos < header_str.len && header_str[pos] >= '0' && header_str[pos] <= '9') {
                            val = val * 10 + (ulong)(header_str[pos] - '0');
                            pos++;
                        }
                        if (n_dims < 4) shape[n_dims] = val;
                        n_dims++;
                    }
                    if (pos < header_str.len) pos++; // skip ']'
                }
            } else if (field_key == "data_offsets") {
                // Parse [start, end]
                if (header_str[pos] == '[') {
                    pos++;
                    while (pos < header_str.len && (header_str[pos] == ' ' || header_str[pos] == ',')) pos++;
                    // Parse start
                    while (pos < header_str.len && header_str[pos] >= '0' && header_str[pos] <= '9') {
                        data_start = data_start * 10 + (ulong)(header_str[pos] - '0');
                        pos++;
                    }
                    while (pos < header_str.len && (header_str[pos] == ' ' || header_str[pos] == ',')) pos++;
                    // Parse end
                    while (pos < header_str.len && header_str[pos] >= '0' && header_str[pos] <= '9') {
                        data_end = data_end * 10 + (ulong)(header_str[pos] - '0');
                        pos++;
                    }
                    while (pos < header_str.len && header_str[pos] != ']') pos++;
                    if (pos < header_str.len) pos++; // skip ']'
                }
            } else {
                // Skip unknown field value
                if (header_str[pos] == '"') {
                    pos++;
                    while (pos < header_str.len && header_str[pos] != '"') pos++;
                    pos++;
                } else if (header_str[pos] == '[') {
                    uint d = 1; pos++;
                    while (pos < header_str.len && d > 0) {
                        if (header_str[pos] == '[') d++;
                        if (header_str[pos] == ']') d--;
                        pos++;
                    }
                } else if (header_str[pos] == '{') {
                    uint d = 1; pos++;
                    while (pos < header_str.len && d > 0) {
                        if (header_str[pos] == '{') d++;
                        if (header_str[pos] == '}') d--;
                        pos++;
                    }
                }
            }
        }

        if (n_tensors < 256) {
            tensors[n_tensors] = {
                .name = key,
                .dtype = dtype,
                .shape = shape,
                .n_dims = n_dims,
                .data_start = data_start,
                .data_end = data_end,
            };
            n_tensors++;
        }
    }

    char* data_base = &data[8 + (usz)header_len];

    io::printfn("Safetensors: %d tensors loaded from %s", n_tensors, path);

    return {
        .tensors = tensors,
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
        f32_data = (float*)mem::calloc(f32_size);
        needs_free = true;
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_f16_to_f32(data_ptr + j * 2);
        }
    } else if (info.dtype == "BF16") {
        f32_data = (float*)mem::calloc(f32_size);
        needs_free = true;
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_bf16_to_f32(data_ptr + j * 2);
        }
    } else {
        io::printfn("Warning: unsupported safetensors dtype '%s' for %s", info.dtype, name);
        return COMPUTE_ERROR~;
    }

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: f32_data,
        data_size: f32_size,
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

fn void SafetensorsFile.close(&self) {
    mem::free(self.tensors);
}
