#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Download stories260K model if not present
if [ ! -f test/models/stories260K.gguf ]; then
    echo "Downloading stories260K.gguf..."
    mkdir -p test/models
    curl -L -o test/models/stories260K.gguf \
        "https://huggingface.co/ggml-org/tiny-llamas/resolve/def3e2dd70df35ecbf6403ea347de4c5977220c1/stories260K.gguf?download=true"
    echo "Downloaded stories260K.gguf ($(du -h test/models/stories260K.gguf | cut -f1))"
fi

# Run tests (--test-noleak: Vulkan/GPU allocations aren't tracked by C3's allocator)
echo ""
echo "Running c3c test..."
c3c test --test-noleak "$@"
