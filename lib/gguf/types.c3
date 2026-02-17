module llm;

import std::io;
import std::core::mem;

const uint GGUF_MAGIC = 0x46554747;

const usz GGUF_ALIGNMENT = 32;

// GGUF metadata value types (0-12, contiguous)
enum GGUFType : uint {
    UINT8,
    INT8,
    UINT16,
    INT16,
    UINT32,
    INT32,
    FLOAT32,
    BOOL,
    STRING,
    ARRAY,
    UINT64,
    INT64,
    FLOAT64,
}

// GGML tensor quantization types (has gaps, use typedef)
typedef GGMLType = inline uint;

const GGMLType GGML_F32    = 0;
const GGMLType GGML_F16    = 1;
const GGMLType GGML_Q4_0   = 2;
const GGMLType GGML_Q4_1   = 3;
const GGMLType GGML_Q5_0   = 6;
const GGMLType GGML_Q5_1   = 7;
const GGMLType GGML_Q8_0   = 8;
const GGMLType GGML_Q8_1   = 9;
const GGMLType GGML_Q2_K   = 10;
const GGMLType GGML_Q3_K   = 11;
const GGMLType GGML_Q4_K   = 12;
const GGMLType GGML_Q5_K   = 13;
const GGMLType GGML_Q6_K   = 14;
const GGMLType GGML_I8     = 24;
const GGMLType GGML_I16    = 25;
const GGMLType GGML_I32    = 26;
const GGMLType GGML_I64    = 27;
const GGMLType GGML_F64    = 28;
const GGMLType GGML_BF16   = 30;

fn usz GGMLType.block_size(self) {
    switch (self) {
        case GGML_F32:    return 1;
        case GGML_F16:    return 1;
        case GGML_Q4_0:   return 32;
        case GGML_Q4_1:   return 32;
        case GGML_Q5_0:   return 32;
        case GGML_Q5_1:   return 32;
        case GGML_Q8_0:   return 32;
        case GGML_Q8_1:   return 32;
        case GGML_Q2_K:   return 256;
        case GGML_Q3_K:   return 256;
        case GGML_Q4_K:   return 256;
        case GGML_Q5_K:   return 256;
        case GGML_Q6_K:   return 256;
        case GGML_I8:     return 1;
        case GGML_I16:    return 1;
        case GGML_I32:    return 1;
        case GGML_I64:    return 1;
        case GGML_F64:    return 1;
        case GGML_BF16:   return 1;
        default:          return 32;
    }
}

fn usz GGMLType.type_size(self) {
    switch (self) {
        case GGML_F32:    return 4;
        case GGML_F16:    return 2;
        case GGML_Q4_0:   return 18;
        case GGML_Q4_1:   return 20;
        case GGML_Q5_0:   return 22;
        case GGML_Q5_1:   return 24;
        case GGML_Q8_0:   return 34;
        case GGML_Q8_1:   return 36;
        case GGML_Q2_K:   return 84;
        case GGML_Q3_K:   return 110;
        case GGML_Q4_K:   return 144;
        case GGML_Q5_K:   return 176;
        case GGML_Q6_K:   return 210;
        case GGML_I8:     return 1;
        case GGML_I16:    return 2;
        case GGML_I32:    return 4;
        case GGML_I64:    return 8;
        case GGML_F64:    return 8;
        case GGML_BF16:   return 2;
        default:          return 0;
    }
}

struct BlockQ8_0 {
    ushort scale;  // f16 scale
    ichar[32] qs;  // quantized values
}

union GGUFMetaValue {
    char u8_val;
    ichar i8_val;
    ushort u16_val;
    short i16_val;
    uint u32_val;
    int i32_val;
    float f32_val;
    bool bool_val;
    String str_val;
    ulong u64_val;
    long i64_val;
    double f64_val;
}

struct GGUFMetaKV {
    String key;
    GGUFType val_type;
    GGUFMetaValue value;
    // For arrays
    GGUFType array_type;
    ulong array_len;
    usz array_data_offset;
}

struct GGUFTensorInfo {
    String name;
    uint n_dims;
    ulong[4] shape;
    GGMLType type;
    ulong offset;
    usz data_size;
}

struct GGUFFile {
    uint version;
    ulong n_tensors;
    ulong n_kv;
    GGUFMetaKV[] metadata;
    GGUFTensorInfo[] tensors;
    char* tensor_data_base;
    char[] raw_data;
}

fn String? GGUFFile.get_string(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == STRING) {
            return kv.value.str_val;
        }
    }
    return FILE_NOT_FOUND~;
}

fn uint? GGUFFile.get_u32(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == UINT32) {
            return kv.value.u32_val;
        }
    }
    return FILE_NOT_FOUND~;
}

fn int? GGUFFile.get_i32(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == INT32) {
            return kv.value.i32_val;
        }
    }
    return FILE_NOT_FOUND~;
}

fn float? GGUFFile.get_f32(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == FLOAT32) {
            return kv.value.f32_val;
        }
    }
    return FILE_NOT_FOUND~;
}

fn GGUFTensorInfo*? GGUFFile.find_tensor(&self, String name) {
    foreach (&t : self.tensors) {
        if (t.name == name) {
            return t;
        }
    }
    return FILE_NOT_FOUND~;
}

fn String[]? GGUFFile.read_string_array(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == ARRAY && kv.array_type == STRING) {
            usz count = (usz)kv.array_len;
            String[] result = mem::new_array(String, count);
            // Create reader at the array data offset
            usz pos = kv.array_data_offset;
            for (usz i = 0; i < count; i++) {
                ulong len = bitcast(*(char[8]*)self.raw_data[pos..], ulong);
                pos += 8;
                usz slen = (usz)len;
                if (slen > 0) {
                    result[i] = (String)self.raw_data[pos .. pos + slen - 1];
                } else {
                    result[i] = "";
                }
                pos += slen;
            }
            return result;
        }
    }
    return FILE_NOT_FOUND~;
}

fn float[]? GGUFFile.read_f32_array(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == ARRAY && kv.array_type == FLOAT32) {
            usz count = (usz)kv.array_len;
            float[] result = mem::new_array(float, count);
            usz pos = kv.array_data_offset;
            for (usz i = 0; i < count; i++) {
                result[i] = bitcast(*(char[4]*)self.raw_data[pos..], float);
                pos += 4;
            }
            return result;
        }
    }
    return FILE_NOT_FOUND~;
}

fn uint[]? GGUFFile.read_u32_array(&self, String key) {
    foreach (&kv : self.metadata) {
        if (kv.key == key && kv.val_type == ARRAY && (kv.array_type == UINT32 || kv.array_type == INT32)) {
            usz count = (usz)kv.array_len;
            uint[] result = mem::new_array(uint, count);
            usz pos = kv.array_data_offset;
            for (usz i = 0; i < count; i++) {
                result[i] = bitcast(*(char[4]*)self.raw_data[pos..], uint);
                pos += 4;
            }
            return result;
        }
    }
    return FILE_NOT_FOUND~;
}
