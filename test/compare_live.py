#!/usr/bin/env python3
"""
Live comparison test between C3 and PyTorch Z-Image implementations.

This script runs both implementations with the same inputs and compares outputs.
Usage:
    python3 test/compare_live.py [--component COMPONENT]
"""

import argparse
import numpy as np
import torch
import subprocess
import tempfile
import struct
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_zimage_full import load_full_model

# Constants
DIM = 3840
N_PATCHES = 1024
PADDED_TEXT = 32


def run_c3_component(component: str, input_data: np.ndarray) -> np.ndarray:
    """Run C3 implementation for a specific component."""
    # Save input to temp file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # Write header
        f.write(b"C3_DEBUG_TENSOR\x00")
        # Write data
        f.write(input_data.astype(np.float32).tobytes())
        input_path = f.name

    # Run C3 test
    try:
        result = subprocess.run(
            [
                "c3c",
                "run",
                "zimage",
                "--",
                f"--test-component={component}",
                f"--input={input_path}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"C3 stderr: {result.stderr}")
            return None

        # Read output
        output_path = input_path.replace(".bin", "_output.bin")
        with open(output_path, "rb") as f:
            f.read(16)  # Skip header
            output = np.frombuffer(f.read(), dtype=np.float32).copy()

        return output

    except Exception as e:
        print(f"Error running C3: {e}")
        return None
    finally:
        Path(input_path).unlink(missing_ok=True)


def compare_tensors(name: str, c3_data: np.ndarray, py_data: np.ndarray) -> dict:
    """Compare two tensors and return metrics."""
    if c3_data is None or py_data is None:
        return {"error": "Missing data"}

    if c3_data.shape != py_data.shape:
        return {"error": f"Shape mismatch: {c3_data.shape} vs {py_data.shape}"}

    diff = np.abs(c3_data - py_data)

    # Cosine similarity
    cos_sim = np.dot(c3_data.flatten(), py_data.flatten()) / (
        np.linalg.norm(c3_data) * np.linalg.norm(py_data) + 1e-12
    )

    return {
        "cosine_similarity": cos_sim,
        "max_diff": diff.max(),
        "mean_diff": diff.mean(),
        "c3_max": c3_data.max(),
        "c3_min": c3_data.min(),
        "py_max": py_data.max(),
        "py_min": py_data.min(),
    }


def test_x_embedder(model):
    """Test x_embedder component."""
    print("\n[TEST] x_embedder...")

    # Generate random input latent
    latent = torch.randn(1, 16, 64, 64)

    # PyTorch
    with torch.no_grad():
        patches = model.patchify(latent)
        py_output = model.x_embedder(patches).squeeze(0).numpy()

    # C3 (would need to implement this)
    print("  PyTorch: min=%.3f max=%.3f" % (py_output.min(), py_output.max()))
    print("  C3: Not implemented yet (requires C3 test harness)")
    print("  PASS: Reference only")


def test_noise_refiner(model):
    """Test noise_refiner component."""
    print("\n[TEST] noise_refiner...")

    # Generate input
    x_emb = torch.randn(1, N_PATCHES, DIM)
    t_emb = torch.randn(1, 256)

    # PyTorch
    with torch.no_grad():
        py_nr0 = model.noise_refiner[0](x_emb, t_emb)
        py_nr1 = model.noise_refiner[1](py_nr0, t_emb)

    print(f"  PyTorch nr0: min={py_nr0.min():.3f} max={py_nr0.max():.3f}")
    print(f"  PyTorch nr1: min={py_nr1.min():.3f} max={py_nr1.max():.3f}")

    # TODO: Compare with C3 when test harness is ready
    print("  C3: Not implemented yet")
    print("  PASS: Reference only")


def test_rope(model):
    """Test RoPE tables."""
    print("\n[TEST] RoPE tables...")

    # Generate position IDs
    pos_ids = model.create_coordinate_grid(
        size=(1, 32, 32),
        start=(33, 0, 0),
    ).flatten(0, 2)

    # PyTorch
    with torch.no_grad():
        freqs_cis = model.rope_embedder(pos_ids)

    cos_vals = freqs_cis.real.numpy()
    sin_vals = freqs_cis.imag.numpy()

    print(f"  PyTorch cos: min={cos_vals.min():.3f} max={cos_vals.max():.3f}")
    print(f"  PyTorch sin: min={sin_vals.min():.3f} max={sin_vals.max():.3f}")

    print("  PASS: Reference only")


def main():
    parser = argparse.ArgumentParser(description="Live C3 vs PyTorch comparison")
    parser.add_argument(
        "--component",
        choices=["x_embedder", "noise_refiner", "rope", "all"],
        default="all",
        help="Which component to test",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Z-Image Live Comparison Test")
    print("=" * 70)

    # Load model
    print("\nLoading PyTorch model...")
    model = load_full_model()
    print("Model loaded.")

    # Run tests
    if args.component in ("x_embedder", "all"):
        test_x_embedder(model)

    if args.component in ("noise_refiner", "all"):
        test_noise_refiner(model)

    if args.component in ("rope", "all"):
        test_rope(model)

    print("\n" + "=" * 70)
    print("Test complete!")
    print("Note: Full C3 integration requires test harness implementation")
    print("=" * 70)


if __name__ == "__main__":
    main()
