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

# Leak detection is on: the suite allocates nothing it does not release, so a
# leak report here is a real regression rather than pre-existing noise. Vulkan
# and GPU allocations are invisible to C3's allocator either way, so this only
# ever polices host-side memory.
echo ""
echo "Running c3c test..."
c3c test "$@"
