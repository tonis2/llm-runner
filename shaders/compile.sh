#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling llm.slang -> llm.spv (all entry points)"
slangc llm.slang -fvk-use-entrypoint-name -target spirv -emit-spirv-directly -o llm.spv

echo "Shader compiled successfully."
