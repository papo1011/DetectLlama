#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
GPU_BACKEND="auto"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 6)"
TARGETS=("DetectLlama" "llama-cli")

usage() {
    cat <<EOF
Usage: $0 [--build-dir DIR] [--gpu auto|cpu|cuda|metal|vulkan] [--jobs N]

Examples:
  $0
  $0 --gpu cuda --jobs 8
  $0 --build-dir build-cuda --gpu cuda
EOF
}

format_duration() {
    local total="$1"
    local minutes=$((total / 60))
    local seconds=$((total % 60))

    if [ "$minutes" -gt 0 ]; then
        printf "%dm %02ds" "$minutes" "$seconds"
    else
        printf "%ds" "$seconds"
    fi
}

run_with_idle_spinner() {
    local idle_seconds=2
    local spin_chars='|/-\'
    local spin_index=0
    local spinner_visible=0
    local last_output
    local now
    local line
    local status
    local tmp_dir
    local fifo
    local cmd_pid

    tmp_dir="$(mktemp -d)"
    fifo="$tmp_dir/output"
    mkfifo "$fifo"

    exec 3<>"$fifo"
    "$@" > "$fifo" 2>&1 &
    cmd_pid="$!"

    last_output="$(date +%s)"

    while kill -0 "$cmd_pid" 2>/dev/null; do
        if IFS= read -r -t 0.2 line <&3; then
            if [ "$spinner_visible" -eq 1 ]; then
                printf "\r\033[K"
                spinner_visible=0
            fi
            printf "%s\n" "$line"
            last_output="$(date +%s)"
        else
            now="$(date +%s)"
            if [ -t 1 ] && [ "$((now - last_output))" -ge "$idle_seconds" ]; then
                printf "\r    waiting... %s" "${spin_chars:$spin_index:1}"
                spin_index=$(((spin_index + 1) % ${#spin_chars}))
                spinner_visible=1
            fi
        fi
    done

    while IFS= read -r -t 0.1 line <&3; do
        if [ "$spinner_visible" -eq 1 ]; then
            printf "\r\033[K"
            spinner_visible=0
        fi
        printf "%s\n" "$line"
    done

    if wait "$cmd_pid"; then
        status=0
    else
        status="$?"
    fi

    if [ "$spinner_visible" -eq 1 ]; then
        printf "\r\033[K"
    fi

    exec 3>&-
    rm -rf "$tmp_dir"
    return "$status"
}

llama_artifact_exists() {
    find "$BUILD_DIR" \( -name 'libllama.*' -o -name 'llama.dll' -o -name 'llama.lib' \) -print -quit 2>/dev/null |
        grep -q .
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU_BACKEND="$2"
            shift 2
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$GPU_BACKEND" in
    auto|cpu|cuda|metal|vulkan) ;;
    *)
        echo "Unsupported GPU backend: $GPU_BACKEND" >&2
        exit 1
        ;;
esac

START_TOTAL="$(date +%s)"

echo "==> Configure"
echo "    build dir: $BUILD_DIR"
echo "    gpu:       $GPU_BACKEND"

START_CONFIGURE="$(date +%s)"
run_with_idle_spinner cmake -B "$BUILD_DIR" . -DDETECT_LLAMA_GPU="$GPU_BACKEND"
END_CONFIGURE="$(date +%s)"

LLAMA_SRC="$BUILD_DIR/_deps/llama_cpp-src"
LLAMA_BUILD="$BUILD_DIR/_deps/llama_cpp-build"
if [ -d "$LLAMA_SRC" ]; then
    echo "==> llama.cpp"
    echo "    source: $LLAMA_SRC"
    if llama_artifact_exists; then
        echo "    mode:   bundled fallback, incremental build"
    else
        echo "    mode:   bundled fallback, first build may take a while"
    fi
else
    echo "==> llama.cpp"
    echo "    mode:   system package"
fi

echo "==> Build"
echo "    targets: ${TARGETS[*]}"
echo "    jobs:   $JOBS"

START_BUILD="$(date +%s)"
run_with_idle_spinner cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" --parallel "$JOBS"
END_BUILD="$(date +%s)"
END_TOTAL="$(date +%s)"

echo "==> Done"
echo "    configure: $(format_duration "$((END_CONFIGURE - START_CONFIGURE))")"
echo "    build:     $(format_duration "$((END_BUILD - START_BUILD))")"
echo "    total:     $(format_duration "$((END_TOTAL - START_TOTAL))")"
