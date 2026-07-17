#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
GPU_BACKEND="auto"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 6)"
TARGETS=("DetectLlama" "DetectLlamaBackend")
LLAMA_MODE=""
LLAMA_PATH=""
# macOS still ships Bash 3.2, whose read -t builtin rejects fractional
# timeouts such as 0.2. Keep these integer-only for portability.
READ_POLL_TIMEOUT=1
READ_DRAIN_TIMEOUT=1

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --build-dir DIR       CMake build directory (default: build)
  --gpu BACKEND         auto, cpu, cuda, metal, or vulkan
  --jobs N, -j N        Parallel build jobs
  --llama-path DIR      Use a local llama.cpp source checkout
  --download-llama      Allow DetectLlama to download and build llama.cpp
  --help, -h            Show this help

Examples:
  $0
  $0 --gpu cuda --jobs 8
  $0 --build-dir build-cuda --gpu cuda
  $0 --llama-path ~/src/llama.cpp
  $0 --download-llama
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
        if IFS= read -r -t "$READ_POLL_TIMEOUT" line <&3; then
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

    while IFS= read -r -t "$READ_DRAIN_TIMEOUT" line <&3; do
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

cmake_cache_value() {
    local key="$1"
    local cache_file="$BUILD_DIR/CMakeCache.txt"

    if [ ! -f "$cache_file" ]; then
        return
    fi

    sed -n "s/^${key}:[^=]*=//p" "$cache_file" | tail -n 1
}

is_llama_source_checkout() {
    local path="$1"
    [ -f "$path/CMakeLists.txt" ] && [ -f "$path/include/llama.h" ]
}

llama_package_exists() {
    local compiler_id="GNU"
    if [ "$(uname -s)" = "Darwin" ]; then
        compiler_id="AppleClang"
    fi

    cmake --find-package \
        -DNAME=llama \
        -DCOMPILER_ID="$compiler_id" \
        -DLANGUAGE=CXX \
        -DMODE=EXIST >/dev/null 2>&1
}

prompt_for_llama_provider() {
    local choice
    local selected_path

    echo
    echo "llama.cpp was not found as an installed CMake package."
    echo "Choose how DetectLlama should continue:"
    echo "  1) Use a local llama.cpp source checkout"
    echo "  2) Download and build llama.cpp"
    echo "  3) Cancel"

    while true; do
        printf "Selection [1-3]: "
        IFS= read -r choice

        case "$choice" in
            1)
                while true; do
                    printf "Path to llama.cpp: "
                    IFS= read -r selected_path
                    case "$selected_path" in
                        "~")
                            selected_path="$HOME"
                            ;;
                        "~/"*)
                            selected_path="$HOME/${selected_path#\~/}"
                            ;;
                    esac

                    if is_llama_source_checkout "$selected_path"; then
                        LLAMA_MODE="local"
                        LLAMA_PATH="$selected_path"
                        return
                    fi
                    echo "That directory is not a llama.cpp source checkout."
                    echo "Expected CMakeLists.txt and include/llama.h."
                done
                ;;
            2)
                LLAMA_MODE="download"
                return
                ;;
            3)
                echo "Build cancelled."
                exit 0
                ;;
            *)
                echo "Enter 1, 2, or 3."
                ;;
        esac
    done
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
        --llama-path)
            if [ "$LLAMA_MODE" = "download" ]; then
                echo "--llama-path and --download-llama cannot be used together." >&2
                exit 1
            fi
            LLAMA_MODE="local"
            LLAMA_PATH="$2"
            shift 2
            ;;
        --download-llama)
            if [ "$LLAMA_MODE" = "local" ]; then
                echo "--llama-path and --download-llama cannot be used together." >&2
                exit 1
            fi
            LLAMA_MODE="download"
            shift
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

if [ "$LLAMA_MODE" = "local" ] && ! is_llama_source_checkout "$LLAMA_PATH"; then
    echo "Not a llama.cpp source checkout: $LLAMA_PATH" >&2
    echo "Expected CMakeLists.txt and include/llama.h." >&2
    exit 1
fi

if [ -z "$LLAMA_MODE" ]; then
    CACHED_LLAMA_PATH="$(cmake_cache_value DETECT_LLAMA_CPP_DIR)"
    CACHED_LLAMA_FETCH="$(cmake_cache_value DETECT_LLAMA_FETCH)"

    if [ -n "$CACHED_LLAMA_PATH" ] && is_llama_source_checkout "$CACHED_LLAMA_PATH"; then
        LLAMA_MODE="local"
        LLAMA_PATH="$CACHED_LLAMA_PATH"
    elif [ "$CACHED_LLAMA_FETCH" = "ON" ] || [ -f "$BUILD_DIR/_deps/llama_cpp-src/CMakeLists.txt" ]; then
        LLAMA_MODE="download"
    elif llama_package_exists; then
        LLAMA_MODE="system"
    elif [ -t 0 ]; then
        prompt_for_llama_provider
    else
        echo "llama.cpp was not found and the build is not interactive." >&2
        echo "Pass --llama-path DIR or --download-llama to choose explicitly." >&2
        exit 1
    fi
fi

START_TOTAL="$(date +%s)"

echo "==> Configure"
echo "    build dir: $BUILD_DIR"
echo "    gpu:       $GPU_BACKEND"

CMAKE_ARGS=(-B "$BUILD_DIR" . -DDETECT_LLAMA_GPU="$GPU_BACKEND")
case "$LLAMA_MODE" in
    local)
        CMAKE_ARGS+=(
            -DDETECT_LLAMA_CPP_DIR="$LLAMA_PATH"
            -DDETECT_LLAMA_FETCH=OFF
        )
        ;;
    download)
        CMAKE_ARGS+=(
            -DDETECT_LLAMA_CPP_DIR=
            -DDETECT_LLAMA_FETCH=ON
        )
        ;;
    system)
        CMAKE_ARGS+=(
            -DDETECT_LLAMA_CPP_DIR=
            -DDETECT_LLAMA_FETCH=OFF
        )
        ;;
esac

START_CONFIGURE="$(date +%s)"
run_with_idle_spinner cmake "${CMAKE_ARGS[@]}"
END_CONFIGURE="$(date +%s)"

LLAMA_PROVIDER="$(cmake_cache_value DETECT_LLAMA_RESOLVED_PROVIDER)"
echo "==> llama.cpp"
case "$LLAMA_PROVIDER" in
    local)
        echo "    mode:   local source checkout"
        echo "    source: $(cmake_cache_value DETECT_LLAMA_CPP_DIR)"
        ;;
    download)
        echo "    source: $BUILD_DIR/_deps/llama_cpp-src"
        if llama_artifact_exists; then
            echo "    mode:   downloaded source, incremental build"
        else
            echo "    mode:   downloaded source, first build may take a while"
        fi
        ;;
    system)
        echo "    mode:   system package"
        ;;
    *)
        echo "    mode:   unknown"
        ;;
esac

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
