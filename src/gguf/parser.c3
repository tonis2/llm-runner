module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;

struct GGUFReader {
    char[] data;
    usz pos;
}

fn char GGUFReader.read_u8(&self) {
    char v = self.data[self.pos];
    self.pos += 1;
    return v;
}

fn ushort GGUFReader.read_u16(&self) {
    ushort v = bitcast(*(char[2]*)self.data[self.pos..], ushort);
    self.pos += 2;
    return v;
}

fn uint GGUFReader.read_u32(&self) {
    uint v = bitcast(*(char[4]*)self.data[self.pos..], uint);
    self.pos += 4;
    return v;
}

fn int GGUFReader.read_i32(&self) {
    int v = bitcast(*(char[4]*)self.data[self.pos..], int);
    self.pos += 4;
    return v;
}

fn ulong GGUFReader.read_u64(&self) {
    ulong v = bitcast(*(char[8]*)self.data[self.pos..], ulong);
    self.pos += 8;
    return v;
}

fn long GGUFReader.read_i64(&self) {
    long v = bitcast(*(char[8]*)self.data[self.pos..], long);
    self.pos += 8;
    return v;
}

fn float GGUFReader.read_f32(&self) {
    float v = bitcast(*(char[4]*)self.data[self.pos..], float);
    self.pos += 4;
    return v;
}

fn double GGUFReader.read_f64(&self) {
    double v = bitcast(*(char[8]*)self.data[self.pos..], double);
    self.pos += 8;
    return v;
}

fn String GGUFReader.read_string(&self) {
    ulong len = self.read_u64();
    usz slen = (usz)len;
    // C3 ranges are inclusive: arr[a..b] gives b-a+1 elements
    String s = (String)self.data[self.pos .. self.pos + slen - 1];
    self.pos += slen;
    return s;
}

fn void GGUFReader.skip_value(&self, GGUFType val_type) {
    switch (val_type) {
        case UINT8:   self.pos += 1;
        case INT8:    self.pos += 1;
        case UINT16:  self.pos += 2;
        case INT16:   self.pos += 2;
        case UINT32:  self.pos += 4;
        case INT32:   self.pos += 4;
        case FLOAT32: self.pos += 4;
        case BOOL:    self.pos += 1;
        case STRING:  self.read_string();
        case UINT64:  self.pos += 8;
        case INT64:   self.pos += 8;
        case FLOAT64: self.pos += 8;
        case ARRAY: {
            GGUFType arr_type = (GGUFType)self.read_u32();
            ulong arr_len = self.read_u64();
            for (ulong i = 0; i < arr_len; i++) {
                self.skip_value(arr_type);
            }
        }
    }
}

fn GGUFMetaKV GGUFReader.read_meta_kv(&self) {
    String key = self.read_string();
    GGUFType val_type = (GGUFType)self.read_u32();

    GGUFMetaKV kv = {
        .key = key,
        .val_type = val_type,
    };

    switch (val_type) {
        case UINT8:   kv.value.u8_val = self.read_u8();
        case INT8:    kv.value.i8_val = (ichar)self.read_u8();
        case UINT16:  kv.value.u16_val = self.read_u16();
        case INT16:   kv.value.i16_val = (short)self.read_u16();
        case UINT32:  kv.value.u32_val = self.read_u32();
        case INT32:   kv.value.i32_val = self.read_i32();
        case FLOAT32: kv.value.f32_val = self.read_f32();
        case BOOL:    kv.value.bool_val = self.read_u8() != 0;
        case STRING:  kv.value.str_val = self.read_string();
        case UINT64:  kv.value.u64_val = self.read_u64();
        case INT64:   kv.value.i64_val = self.read_i64();
        case FLOAT64: kv.value.f64_val = self.read_f64();
        case ARRAY: {
            kv.array_type = (GGUFType)self.read_u32();
            kv.array_len = self.read_u64();
            kv.array_data_offset = self.pos;
            // Skip array contents for now - we read scalars by key
            for (ulong i = 0; i < kv.array_len; i++) {
                self.skip_value(kv.array_type);
            }
        }
    }

    return kv;
}

fn GGUFTensorInfo GGUFReader.read_tensor_info(&self) {
    String name = self.read_string();
    uint n_dims = self.read_u32();
    ulong[4] shape;
    for (uint i = 0; i < n_dims; i++) {
        shape[i] = self.read_u64();
    }
    GGMLType dtype = (GGMLType)self.read_u32();
    ulong offset = self.read_u64();

    // Compute data size
    usz n_elements = 1;
    for (uint i = 0; i < n_dims; i++) {
        n_elements *= (usz)shape[i];
    }
    usz block_size = dtype.block_size();
    usz type_size = dtype.type_size();
    usz n_blocks = (n_elements + block_size - 1) / block_size;
    usz data_size = n_blocks * type_size;

    return {
        .name = name,
        .n_dims = n_dims,
        .shape = shape,
        .type = dtype,
        .offset = offset,
        .data_size = data_size,
    };
}

fn usz align_offset(usz offset, usz alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

fn GGUFFile? gguf_parse(char[] data) {
    GGUFReader reader = { .data = data, .pos = 0 };

    uint magic = reader.read_u32();
    if (magic != GGUF_MAGIC) {
        io::printfn("Invalid GGUF magic: 0x%08x (expected 0x%08x)", magic, GGUF_MAGIC);
        return FILE_NOT_FOUND~;
    }

    uint version = reader.read_u32();
    if (version < 2 || version > 3) {
        io::printfn("Unsupported GGUF version: %d", version);
        return FILE_NOT_FOUND~;
    }

    ulong n_tensors = reader.read_u64();
    ulong n_kv = reader.read_u64();

    io::printfn("GGUF v%d: %d metadata entries, %d tensors", version, n_kv, n_tensors);

    // Allocate metadata array
    GGUFMetaKV[] metadata = mem::new_array(GGUFMetaKV, (usz)n_kv);
    for (ulong i = 0; i < n_kv; i++) {
        metadata[(usz)i] = reader.read_meta_kv();
    }

    // Allocate tensor info array
    GGUFTensorInfo[] tensors = mem::new_array(GGUFTensorInfo, (usz)n_tensors);
    for (ulong i = 0; i < n_tensors; i++) {
        tensors[(usz)i] = reader.read_tensor_info();
    }

    // Align to tensor data
    usz tensor_data_offset = align_offset(reader.pos, GGUF_ALIGNMENT);
    char* tensor_data_base = &data[tensor_data_offset];

    return {
        .version = version,
        .n_tensors = n_tensors,
        .n_kv = n_kv,
        .metadata = metadata,
        .tensors = tensors,
        .tensor_data_base = tensor_data_base,
        .raw_data = data,
    };
}

fn void GGUFFile.free(&self) {
    if (self.metadata.len > 0) {
        mem::free(self.metadata);
    }
    if (self.tensors.len > 0) {
        mem::free(self.tensors);
    }
}

fn void gguf_print_info(GGUFFile* gf) {
    io::printfn("\n=== GGUF Metadata ===");
    foreach (&kv : gf.metadata) {
        switch (kv.val_type) {
            case UINT32:  io::printfn("  %s = %d (u32)", kv.key, kv.value.u32_val);
            case INT32:   io::printfn("  %s = %d (i32)", kv.key, kv.value.i32_val);
            case FLOAT32: io::printfn("  %s = %f (f32)", kv.key, kv.value.f32_val);
            case STRING:  io::printfn("  %s = %s (str)", kv.key, kv.value.str_val);
            case BOOL:    io::printfn("  %s = %s (bool)", kv.key, kv.value.bool_val);
            case UINT8:   io::printfn("  %s = %d (u8)", kv.key, kv.value.u8_val);
            case ARRAY:   io::printfn("  %s = [array of %d]", kv.key, kv.array_len);
            default:      io::printfn("  %s = <type %d>", kv.key, kv.val_type.ordinal);
        }
    }
    io::printfn("\n=== GGUF Tensors (%d) ===", gf.n_tensors);
    foreach (&t : gf.tensors) {
        io::printf("  %s: [", t.name);
        for (uint d = 0; d < t.n_dims; d++) {
            if (d > 0) io::printf(", ");
            io::printf("%d", t.shape[d]);
        }
        io::printfn("] type=%d size=%d bytes", t.type, t.data_size);
    }
}
