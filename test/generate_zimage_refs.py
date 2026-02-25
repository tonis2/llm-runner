#!/usr/bin/env python3
"""
Generate reference data for Z-Image C3 tests.

This script runs the PyTorch implementation and saves intermediate outputs
that C3 tests can compare against.

Usage:
    python3 test/generate_zimage_refs.py [--component COMPONENT]

Components:
    - x_embedder: x_embedder output
    - noise_refiner: noise_refiner layer outputs
    - rope: RoPE cos/sin tables
    - main_layer0: Main DiT layer 0 output
    - full: Full pipeline output
    - all: Generate all reference data (default)
"""

import argparse
import struct
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_zimage_full import load_full_model

# Constants
DIM = 3840
HEADS = 30
HEAD_DIM = 128
N_PATCHES = 1024
PADDED_TEXT = 32
TOTAL_SEQ = N_PATCHES + PADDED_TEXT
DIT_PATCH_DIM = 64
DIT_LATENT_CHANNELS = 16

REF_DIR = Path(__file__).parent / "zimage_refs"


def write_tensor(path: Path, data: np.ndarray):
    """Write tensor in C3 debug format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # Header
        f.write(b"C3_DEBUG_TENSOR\x00")
        # Data
        f.write(data.astype(np.float32).tobytes())
    print(f"  Written: {path} ({data.shape})")


def generate_x_embedder_ref(model):
    """Generate x_embedder reference output."""
    print("\n[GEN] x_embedder reference...")

    # Create dummy latent
    latent = torch.randn(1, DIT_LATENT_CHANNELS, 64, 64)

    # Run through x_embedder
    with torch.no_grad():
        patches = model.patchify(latent)  # [1, 1024, 64]
        x_emb = model.x_embedder(patches)  # [1, 1024, 3840]

    output = x_emb.squeeze(0).numpy()  # [1024, 3840]
    write_tensor(REF_DIR / "x_embedder_output.bin", output)

    print(
        f"  Stats: min={output.min():.3f} max={output.max():.3f} std={output.std():.3f}"
    )


def generate_noise_refiner_ref(model):
    """Generate noise_refiner layer outputs."""
    print("\n[GEN] noise_refiner reference...")

    # Load or create input
    x_emb_path = REF_DIR / "x_embedder_output.bin"
    if x_emb_path.exists():
        with open(x_emb_path, "rb") as f:
            f.read(16)  # Skip header
            x_emb = torch.from_numpy(
                np.frombuffer(f.read(), dtype=np.float32).copy()
            ).reshape(1, N_PATCHES, DIM)
    else:
        x_emb = torch.randn(1, N_PATCHES, DIM)

    # Create timestep embedding
    t_emb = torch.randn(1, 256)  # Simplified

    # Run through noise_refiner layers
    with torch.no_grad():
        # Layer 0
        nr0_out = model.noise_refiner[0](x_emb, t_emb)
        write_tensor(REF_DIR / "nr0_output.bin", nr0_out.squeeze(0).numpy())
        print(f"  Layer 0: min={nr0_out.min():.3f} max={nr0_out.max():.3f}")

        # Layer 1
        nr1_out = model.noise_refiner[1](nr0_out, t_emb)
        write_tensor(REF_DIR / "nr1_output.bin", nr1_out.squeeze(0).numpy())
        print(f"  Layer 1: min={nr1_out.min():.3f} max={nr1_out.max():.3f}")

    # Save pre-noise-refiner for value range tests
    write_tensor(REF_DIR / "pre_noise_refiner.bin", x_emb.squeeze(0).numpy())


def generate_rope_ref(model):
    """Generate RoPE cos/sin tables."""
    print("\n[GEN] RoPE tables...")

    # Noise refiner RoPE (image-only)
    image_pos_ids = model.create_coordinate_grid(
        size=(1, 32, 32),
        start=(PADDED_TEXT + 1, 0, 0),  # (33, 0, 0)
    ).flatten(0, 2)  # [1024, 3]

    freqs_cis = model.rope_embedder(image_pos_ids)  # [1024, 64] complex

    cos_vals = freqs_cis.real.numpy()
    sin_vals = freqs_cis.imag.numpy()

    write_tensor(REF_DIR / "rope_cos_nr.bin", cos_vals)
    write_tensor(REF_DIR / "rope_sin_nr.bin", sin_vals)

    print(f"  Noise refiner cos: min={cos_vals.min():.3f} max={cos_vals.max():.3f}")
    print(f"  Noise refiner sin: min={sin_vals.min():.3f} max={sin_vals.max():.3f}")

    # Main layer RoPE (full sequence)
    # Image positions
    img_pos_ids = model.create_coordinate_grid(
        size=(1, 32, 32),
        start=(PADDED_TEXT + 1, 0, 0),
    ).flatten(0, 2)  # [1024, 3]

    # Text positions
    txt_pos_ids = model.create_coordinate_grid(
        size=(PADDED_TEXT, 1, 1),
        start=(1, 0, 0),
    ).flatten(0, 2)  # [32, 3]

    # Combined
    img_freqs = model.rope_embedder(img_pos_ids)
    txt_freqs = model.rope_embedder(txt_pos_ids)
    unified_freqs = torch.cat([img_freqs, txt_freqs], dim=0)

    write_tensor(REF_DIR / "rope_cos_main.bin", unified_freqs.real.numpy())
    write_tensor(REF_DIR / "rope_sin_main.bin", unified_freqs.imag.numpy())
    print(f"  Main layer: shape={unified_freqs.shape}")


def generate_main_layer0_ref(model):
    """Generate main DiT layer 0 reference."""
    print("\n[GEN] main layer 0 reference...")

    # Create joint hidden (image + text)
    joint = torch.randn(1, TOTAL_SEQ, DIM)
    t_emb = torch.randn(1, 256)

    with torch.no_grad():
        output = model.layers[0](joint, t_emb)

    write_tensor(REF_DIR / "main_layer0_output.bin", output.squeeze(0).numpy())
    print(f"  Stats: min={output.min():.3f} max={output.max():.3f}")


def generate_full_pipeline_ref(model):
    """Generate full pipeline reference."""
    print("\n[GEN] full pipeline reference...")

    # Create input latent
    latent = torch.randn(1, DIT_LATENT_CHANNELS, 64, 64)
    t_emb = torch.randn(1, 256)
    cap_feats = torch.randn(1, 13, 2048)  # Text features
    cap_mask = torch.ones(1, 13, dtype=torch.bool)

    with torch.no_grad():
        # Forward through DiT
        output = model(
            latent,
            t_emb,
            cap_feats,
            cap_mask,
        )

    write_tensor(REF_DIR / "final_latent.bin", output.squeeze(0).numpy())
    print(
        f"  Stats: min={output.min():.3f} max={output.max():.3f} std={output.std():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Z-Image reference data")
    parser.add_argument(
        "--component",
        choices=["x_embedder", "noise_refiner", "rope", "main_layer0", "full", "all"],
        default="all",
        help="Which component to generate (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Z-Image Reference Data Generator")
    print("=" * 70)

    # Create output directory
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading PyTorch model...")
    model = load_full_model()
    print("Model loaded.")

    # Generate requested components
    if args.component in ("x_embedder", "all"):
        generate_x_embedder_ref(model)

    if args.component in ("noise_refiner", "all"):
        generate_noise_refiner_ref(model)

    if args.component in ("rope", "all"):
        generate_rope_ref(model)

    if args.component in ("main_layer0", "all"):
        generate_main_layer0_ref(model)

    if args.component in ("full", "all"):
        generate_full_pipeline_ref(model)

    print("\n" + "=" * 70)
    print("Reference data generation complete!")
    print(f"Output directory: {REF_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
