# Z-Image Test Suite

This directory contains comprehensive tests for the Z-Image diffusion model implementation to prevent regression and help debug issues.

## Overview

The test suite provides:
- **Unit tests** for individual components (x_embedder, noise_refiner, RoPE, etc.)
- **Reference data generation** from PyTorch implementation
- **Value range checks** to detect explosions
- **Cosine similarity comparisons** against PyTorch

## Quick Start

```bash
# Generate reference data from PyTorch
cd /home/tonis/Documents/c3/llm-runner
python3 test/generate_zimage_refs.py

# Run all Z-Image tests
c3c test --test-filter test_zimage_all

# Run specific test
c3c test --test-filter test_zimage_x_embedder
c3c test --test-filter test_zimage_noise_refiner
c3c test --test-filter test_zimage_rope_tables
```

## Test Files

### C3 Tests (`zimage_component_test.c3`)

Located in `test/zimage_component_test.c3`:

1. **`test_zimage_x_embedder`** - Validates x_embedder output matches PyTorch
2. **`test_zimage_noise_refiner`** - Tests noise_refiner layer outputs  
3. **`test_zimage_rope_tables`** - Verifies RoPE cos/sin tables
4. **`test_zimage_value_ranges`** - Detects value explosions
5. **`test_zimage_main_layer0`** - Tests first main DiT layer
6. **`test_zimage_full_pipeline`** - End-to-end integration test

### Python Reference Generator (`generate_zimage_refs.py`)

Generates reference data from PyTorch:

```bash
# Generate all reference data
python3 test/generate_zimage_refs.py

# Generate specific component
python3 test/generate_zimage_refs.py --component x_embedder
python3 test/generate_zimage_refs.py --component noise_refiner
python3 test/generate_zimage_refs.py --component rope
```

### Live Comparison (`compare_live.py`)

Compares C3 and PyTorch outputs in real-time:

```bash
python3 test/compare_live.py --component all
```

## Reference Data Format

Reference data is stored in `test/zimage_refs/` as binary files with format:
- Header: `C3_DEBUG_TENSOR\x00` (16 bytes)
- Data: Raw float32 values

Files:
- `x_embedder_output.bin` - [1024, 3840]
- `nr0_output.bin` - [1024, 3840]
- `nr1_output.bin` - [1024, 3840]
- `rope_cos_nr.bin` - [1024, 64]
- `rope_sin_nr.bin` - [1024, 64]
- `pre_noise_refiner.bin` - [1024, 3840]
- `joint_hidden.bin` - [1056, 3840]
- `main_layer0_output.bin` - [1056, 3840]
- `final_latent.bin` - [16, 64, 64]

## Expected Value Ranges

Based on debugging analysis:

| Stage | Expected Range | Notes |
|-------|---------------|-------|
| x_embedder output | ±8 | Verified correct |
| After norm+scale | ±45 | 6-7x amplification |
| Q/K after matmul | ±320 | Before head norm |
| Q/K after head norm | ±8 | Normalized |
| Q/K after RoPE | ±8 | Preserved magnitude |
| After noise_refiner | ±27 | 4x total amplification |
| Main layers | ±100 | Watch for explosions |
| Final latent | ±10, std~1 | Should be roughly normal |

## Cosine Similarity Thresholds

- **> 0.99**: Excellent match (verified correct)
- **0.95-0.99**: Minor divergence (acceptable)
- **< 0.95**: Significant divergence (investigate)
- **< 0.50**: Major bug (fix required)

## Current Status

✅ **Verified Correct:**
- x_embedder (cos_sim ~1.0)
- noise_refiner matmul/norm (cos_sim ~1.0)
- noise_refiner RoPE (cos_sim 1.0)

⚠️ **Partial Divergence:**
- Full noise_refiner: cos_sim 0.956
- Main layers: Investigating

❌ **To Be Verified:**
- FFN computation
- Full pipeline output

## Adding New Tests

To add a test for a new component:

1. Add reference generation to `generate_zimage_refs.py`:
```python
def generate_my_component_ref(model):
    # Generate PyTorch output
    with torch.no_grad():
        output = model.my_component(input)
    write_tensor(REF_DIR / "my_component.bin", output.numpy())
```

2. Add C3 test to `zimage_component_test.c3`:
```c3
fn void test_zimage_my_component() @test {
    io::printfn("\n[TEST] my_component...");
    
    float* ref = mem::new_array(float, expected_size);
    defer mem::free(ref);
    
    load_reference_tensor(ZIMAGE_REF_DIR.concat("my_component.bin"), ref, expected_size)!;
    
    // Run C3 implementation
    // float* c3_output = run_my_component();
    
    // Compare
    float cos_sim = compute_cosine_similarity(c3_output, ref, expected_size);
    test::@check(cos_sim > COSINE_SIM_THRESHOLD, 
        "cos_sim %.4f < %.2f", cos_sim, COSINE_SIM_THRESHOLD);
}
```

3. Run test:
```bash
c3c test --test-filter test_zimage_my_component
```

## Debugging Failed Tests

When a test fails:

1. Check value ranges - are they reasonable?
2. Compute cosine similarity with reference
3. Compare specific indices with max difference
4. Run live comparison: `python3 test/compare_live.py`
5. Enable debug output in C3 code
6. Visualize with Python analysis scripts

## Tips

- Always regenerate reference data after model weight updates
- Run tests before committing changes
- Value explosion tests catch bugs early
- Cosine similarity > 0.99 indicates correct computation
- Tests run quickly (~1-2 seconds) for fast iteration
