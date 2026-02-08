#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENTRIES=(
    embedding
    rmsnorm
    silu
    matmul
    matmul_q8
    matmul_q5k
    matmul_q6k
    rope
    softmax
    attention
    residual_add
    elemwise_mul
)

for entry in "${ENTRIES[@]}"; do
    echo "Compiling llm.slang -entry ${entry} -> ${entry}.spv"
    slangc llm.slang -entry "$entry" -stage compute -target spirv -o "${entry}.spv"
done

echo "All shaders compiled successfully."
