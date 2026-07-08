#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
EXE_NAME="DetectLlama"

USE_GPU="${USE_GPU:-auto}"
N_CTX="${N_CTX:-2048}"
N_BATCH="${N_BATCH:-2048}"
DETECT_LLAMA_DRY_RUN="${DETECT_LLAMA_DRY_RUN:-0}"

usage() {
    cat <<EOF
Usage: $0

Model selection is automatic. If the selected GGUF is not already in the
llama.cpp cache, the TUI downloads it anonymously from the public Hugging Face
resolve URL.
EOF
}

to_lower() {
    printf "%s" "$1" | tr '[:upper:]' '[:lower:]'
}

detect_accelerator() {
    local os_name
    local arch_name

    os_name="$(uname -s)"
    arch_name="$(uname -m)"

    if [ "$os_name" = "Darwin" ] && [ "$arch_name" = "arm64" ]; then
        echo "apple-unified"
    elif command -v nvidia-smi >/dev/null 2>&1 &&
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
            awk '$1 + 0 >= 6000 { found=1 } END { exit found ? 0 : 1 }'; then
        echo "nvidia"
    else
        echo "cpu"
    fi
}

if [ "$#" -ne 0 ]; then
    usage >&2
    exit 1
fi

EXECUTABLE="$BUILD_DIR/$EXE_NAME"
if [ ! -x "$EXECUTABLE" ] && [ "$DETECT_LLAMA_DRY_RUN" != "1" ]; then
    echo "Executable not found: $EXECUTABLE. Build first with ./scripts/build.sh" >&2
    exit 1
fi

GPU_ARGS=()
USE_GPU_LC="$(to_lower "$USE_GPU")"
ACCELERATOR="$(detect_accelerator)"
if [ "$USE_GPU_LC" = "1" ] || [ "$USE_GPU_LC" = "true" ] || { [ "$USE_GPU_LC" = "auto" ] && [ "$ACCELERATOR" != "cpu" ]; }; then
    GPU_ARGS=(--gpu)
fi

CMD=("$EXECUTABLE" --target-tps "${TARGET_TOKENS_PER_SEC:-30}" -c "$N_CTX" -b "$N_BATCH" "${GPU_ARGS[@]}")

if [ "$DETECT_LLAMA_DRY_RUN" = "1" ]; then
    printf "Dry run command:"
    printf " %q" "${CMD[@]}"
    printf "\n"
    exit 0
fi

"${CMD[@]}"
