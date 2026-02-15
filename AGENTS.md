# Agent Guidelines for LLM Runner

## Build Commands

```bash
# Build the entire project
c3c build

# Build specific target (zimage executable)
c3c build zimage

# Run zimage example
c3c run zimage -- --config zimage.json

# Compile shaders (required before building)
c3c build shaders

# Run all tests
c3c test

# Run a single test by name
c3c test --test-filter test_stories260k_forward

# Run with verbose output
c3c test --verbose

# Clean build
c3c clean

# Build with optimization (check project.json for opt settings)
c3c build -O2
```

## Project Structure

```
lib/                 # Core library source code
├── gguf/           # GGUF file format parsing
├── model/          # ML model implementations (tensor, kernels, diffusion)
├── pipelines/      # High-level inference pipelines
├── vulkan/         # Vulkan compute context
└── shaders/        # GPU compute shaders (Slang)

examples/           # Example applications
└── z-image/        # Z-Image generation example

test/              # Test suite
├── inference_test.c3
├── tokenizer_test.c3
└── models/         # Test model files

dependencies/      # External C3 libraries
```

## Code Style Guidelines

### Module Structure
```c3
module llm;                              // Top of file

// Standard library imports first
import std::io;
import std::core::mem;

// External dependencies next
import vk;

// Internal imports last
import llm_text;
```

### Naming Conventions
- **Types (structs, enums)**: PascalCase (`Tensor`, `DeviceContext`, `GGMLType`)
- **Functions**: snake_case (`create_tensor`, `load_model`, `compute_barrier`)
- **Variables**: snake_case (`gpu_buffer`, `n_dims`, `size_bytes`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_SEQ_LEN`, `DIT_PATCH_SIZE`)
- **Module names**: lowercase (`llm`, `llm_text`, `llm::diffusers`)

### Type System
```c3
// Use optional types for fallible operations
fn Tensor? create_tensor(...) { ... }

// Use faultdef for error handling
faultdef COMPUTE_ERROR, FILE_NOT_FOUND, INVALID_INPUT;

// Return fault on error
if (failed) return COMPUTE_ERROR~;

// Unwrap with !! (asserts success) or ? (returns error)
Tensor t = create_tensor(...)!!;
Tensor t = create_tensor(...)?;  // Propagates error

// Use defer for cleanup
defer t.free();
defer ctx.free();
```

### Structs and Methods
```c3
struct Tensor {
    vk::Memory gpu_buffer;
    GGMLType dtype;
    uint n_dims;
    ulong[4] shape;
}

// Methods use &self or & for self parameter
fn void Tensor.free(&self) {
    self.gpu_buffer.free();
}

// Constructor-like functions return optional
fn Tensor? create_tensor(DeviceContext* ctx, ...) { ... }
```

### Code Formatting
- **Indentation**: 4 spaces
- **Braces**: Opening brace on same line
- **Line length**: ~100 characters soft limit
- **Blank lines**: Between logical sections

```c3
fn void process_data(DeviceContext* ctx, Tensor* input) {
    // Validate inputs
    if (input.size_bytes == 0) return;
    
    // Process data
    for (uint i = 0; i < input.n_dims; i++) {
        do_something();
    }
    
    // Cleanup happens via defer
}
```

### Error Handling
```c3
// Prefer fault-based error handling
fn void? risky_operation() {
    if (condition) return MY_FAULT~;
    return;  // Success
}

// Use try/catch with if-try shorthand
if (try result = fallible_call()) {
    use_result(result);
}

// Or propagate with ?
fn void? wrapper() {
    fallible_call()?;
}
```

### Memory Management
```c3
// Use defer for cleanup - placed right after allocation
Buffer buf = allocate_buffer()!!;
defer buf.free();

// For heap allocations
char[] data = mem::new_array(char, size);
defer mem::free(data);

// Memory allocator pattern
vk::Memory gpu_buffer = vk::new_buffer(
    allocator: &ctx.allocator,
    usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
    properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    data: data_ptr,
    data_size: size,
)!!;
defer gpu_buffer.free();
```

### Testing
```c3
import std::core::test;

fn void my_test() @test {
    // Arrange
    int input = 5;
    
    // Act
    int result = process(input);
    
    // Assert
    test::eq(10, result);
    test::@check(result > 0, "result should be positive");
}
```

## Important Patterns

### Pipeline Architecture
The codebase follows a pipeline pattern similar to PyTorch Diffusers:
- `lib/pipelines/` - High-level orchestration
- `lib/model/` - Individual model components (DiT, VAE, UNet)
- Separate concerns: pipelines coordinate, models compute

### Vulkan Compute
- Use `begin_compute(cmd)` / `submit_and_wait(ctx)` blocks
- `compute_barrier(cmd)` between dependent operations
- `dispatch_kernel()` for GPU shader execution

### Import Guidelines
- Group imports: std library → external deps → internal
- Use fully qualified names for cross-module refs: `llm::diffusers::DiTModel`
- Keep module names lowercase and descriptive

## File Organization

- One major struct per file typically
- Related functions grouped together
- Fault definitions at top of module
- Constants defined near usage or in dedicated config files

## Common Types

- `usz` - usize (unsigned size)
- `char[]` - Byte arrays
- `String` - String slices
- `ulong[4]` - Shape arrays for tensors
- `vk::CommandBuffer` - Vulkan command buffers
- `vk::Memory` - GPU memory handles
