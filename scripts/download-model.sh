#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"

DEFAULT_MODEL_REPO="maddes8cht/tiiuae-falcon-7b-instruct-gguf"
TARGET_TOKENS_PER_SEC_DEFAULT="30"

DRY_RUN=0
PRINT_PATH_ONLY=0

usage() {
    cat <<EOF
Usage: $0 [--build-dir DIR] [--dry-run] [--print-path]

Downloads the selected Falcon 7B GGUF with llama-cli -hf into the standard
llama.cpp/Hugging Face cache.

Environment knobs:
  MODEL_REPO=<huggingface repo>
  TARGET_TOKENS_PER_SEC=30
  LLAMA_CLI_BIN=/path/to/llama-cli
  HF_TOKEN=<optional token>
  N_CTX=2048
  N_BATCH=2048
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --print-path)
            PRINT_PATH_ONLY=1
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

MODEL_REPO="${MODEL_REPO:-$DEFAULT_MODEL_REPO}"
TARGET_TOKENS_PER_SEC="${TARGET_TOKENS_PER_SEC:-$TARGET_TOKENS_PER_SEC_DEFAULT}"
N_CTX="${N_CTX:-2048}"
N_BATCH="${N_BATCH:-2048}"

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf "%s" "$value"
}

to_lower() {
    printf "%s" "$1" | tr '[:upper:]' '[:lower:]'
}

cpu_cores() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1
}

linux_mem_mb() {
    local key="$1"
    awk -v key="$key" '$1 == key ":" { print int($2 / 1024); found=1 } END { if (!found) print 0 }' /proc/meminfo 2>/dev/null
}

mac_total_mem_mb() {
    local bytes
    bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    echo $((bytes / 1024 / 1024))
}

mac_available_mem_mb() {
    local page_size
    page_size="$(pagesize 2>/dev/null || echo 4096)"
    vm_stat 2>/dev/null | awk -v page_size="$page_size" '
        /Pages free/ { free=$3 }
        /Pages inactive/ { inactive=$3 }
        /Pages speculative/ { speculative=$3 }
        END {
            gsub("\\.", "", free)
            gsub("\\.", "", inactive)
            gsub("\\.", "", speculative)
            print int((free + inactive + speculative) * page_size / 1024 / 1024)
        }'
}

detect_nvidia_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null |
        awk -F',' '
            {
                name=$1
                total=$2 + 0
                free=$3 + 0
                gsub(/^[ \t]+|[ \t]+$/, "", name)
                if (free > best_free) {
                    best_name=name
                    best_total=total
                    best_free=free
                }
            }
            END {
                if (best_free > 0) {
                    printf "%s|%d|%d\n", best_name, best_total, best_free
                }
            }'
}

detect_apple_gpu_name() {
    if [ "$(uname -s)" != "Darwin" ]; then
        return 1
    fi

    system_profiler SPHardwareDataType SPDisplaysDataType 2>/dev/null |
        awk -F': ' '
            /Chip:/ && chip == "" { chip=$2 }
            /Chipset Model:/ && gpu == "" { gpu=$2 }
            END {
                if (gpu != "") print gpu
                else if (chip != "") print chip
            }'
}

quant_filename() {
    case "$1" in
        Q4_0) printf "ggml-tiiuae-falcon-7b-instruct-Q4_0.gguf" ;;
        Q4_K_M) printf "tiiuae-falcon-7b-instruct-Q4_K_M.gguf" ;;
        Q5_K_M) printf "tiiuae-falcon-7b-instruct-Q5_K_M.gguf" ;;
        Q6_K) printf "tiiuae-falcon-7b-instruct-Q6_K.gguf" ;;
        Q8_0) printf "ggml-tiiuae-falcon-7b-instruct-Q8_0.gguf" ;;
        *) return 1 ;;
    esac
}

quant_size_mb() {
    case "$1" in
        Q4_0) echo 3900 ;;
        Q4_K_M) echo 4100 ;;
        Q5_K_M) echo 5100 ;;
        Q6_K) echo 5900 ;;
        Q8_0) echo 7600 ;;
        *) echo 0 ;;
    esac
}

quant_rank() {
    case "$1" in
        Q4_0) echo 1 ;;
        Q4_K_M) echo 2 ;;
        Q5_K_M) echo 3 ;;
        Q6_K) echo 4 ;;
        Q8_0) echo 5 ;;
        *) echo 0 ;;
    esac
}

quant_by_rank() {
    case "$1" in
        1) echo "Q4_0" ;;
        2) echo "Q4_K_M" ;;
        3) echo "Q5_K_M" ;;
        4) echo "Q6_K" ;;
        5) echo "Q8_0" ;;
        *) return 1 ;;
    esac
}

estimate_runtime_overhead_mb() {
    local ctx="$1"
    local overhead=1536
    if [ "$ctx" -gt 2048 ]; then
        overhead=$((overhead + ((ctx - 2048 + 2047) / 2048) * 512))
    fi
    echo "$overhead"
}

preferred_quant_for_hardware() {
    local accelerator="$1"
    local gpu_name="$2"
    local gpu_free_mb="$3"
    local total_ram_mb="$4"
    local cores="$5"
    local name_lc

    name_lc="$(to_lower "$gpu_name")"

    if [ "$accelerator" = "nvidia" ]; then
        if [ "$gpu_free_mb" -ge 18000 ]; then
            echo "Q8_0"
        elif [ "$gpu_free_mb" -ge 12000 ]; then
            echo "Q6_K"
        elif [ "$gpu_free_mb" -ge 8500 ]; then
            echo "Q5_K_M"
        elif [ "$gpu_free_mb" -ge 6000 ]; then
            echo "Q4_K_M"
        else
            echo "Q4_0"
        fi
        return
    fi

    if [ "$accelerator" = "apple-unified" ]; then
        if printf "%s" "$name_lc" | grep -Eq 'ultra|max'; then
            if [ "$total_ram_mb" -ge 64000 ]; then
                echo "Q6_K"
            else
                echo "Q5_K_M"
            fi
        elif printf "%s" "$name_lc" | grep -q 'pro'; then
            if [ "$total_ram_mb" -ge 32000 ]; then
                echo "Q5_K_M"
            else
                echo "Q4_K_M"
            fi
        elif [ "$total_ram_mb" -ge 32000 ]; then
            echo "Q5_K_M"
        elif [ "$total_ram_mb" -ge 16000 ]; then
            echo "Q4_K_M"
        else
            echo "Q4_0"
        fi
        return
    fi

    if [ "$cores" -ge 16 ] && [ "$total_ram_mb" -ge 32000 ]; then
        echo "Q4_K_M"
    else
        echo "Q4_0"
    fi
}

apply_speed_target() {
    local quant="$1"
    local target_tps="$2"
    local rank

    rank="$(quant_rank "$quant")"
    if [ "$target_tps" -ge 45 ]; then
        rank=$((rank - 1))
    elif [ "$target_tps" -le 20 ]; then
        rank=$((rank + 1))
    fi

    if [ "$rank" -lt 1 ]; then
        rank=1
    elif [ "$rank" -gt 5 ]; then
        rank=5
    fi

    quant_by_rank "$rank"
}

fits_quant() {
    local quant="$1"
    local memory_pool_mb="$2"
    local disk_free_mb="$3"
    local overhead_mb="$4"
    local size_mb

    size_mb="$(quant_size_mb "$quant")"
    [ "$memory_pool_mb" -ge "$((size_mb + overhead_mb))" ] && [ "$disk_free_mb" -ge "$((size_mb + 1024))" ]
}

select_quant() {
    local preferred="$1"
    local memory_pool_mb="$2"
    local disk_free_mb="$3"
    local overhead_mb="$4"
    local rank
    local quant

    rank="$(quant_rank "$preferred")"
    while [ "$rank" -ge 1 ]; do
        quant="$(quant_by_rank "$rank")"
        if fits_quant "$quant" "$memory_pool_mb" "$disk_free_mb" "$overhead_mb"; then
            echo "$quant"
            return 0
        fi
        rank=$((rank - 1))
    done

    return 1
}

hf_cache_dir() {
    if [ -n "${LLAMA_CACHE:-}" ]; then
        printf "%s" "$LLAMA_CACHE"
    elif [ -n "${HF_HUB_CACHE:-}" ]; then
        printf "%s" "$HF_HUB_CACHE"
    elif [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
        printf "%s" "$HUGGINGFACE_HUB_CACHE"
    elif [ -n "${HF_HOME:-}" ]; then
        printf "%s/hub" "$HF_HOME"
    elif [ -n "${XDG_CACHE_HOME:-}" ]; then
        printf "%s/huggingface/hub" "$XDG_CACHE_HOME"
    else
        printf "%s/.cache/huggingface/hub" "$HOME"
    fi
}

hf_repo_cache_folder() {
    printf "models--%s" "$1" | sed 's#/#--#g'
}

cached_model_path() {
    local repo="$1"
    local filename="$2"
    local cache_dir
    local repo_dir
    local ref_file
    local commit
    local path

    cache_dir="$(hf_cache_dir)"
    repo_dir="$cache_dir/$(hf_repo_cache_folder "$repo")"
    ref_file="$repo_dir/refs/main"

    if [ -f "$ref_file" ]; then
        commit="$(tr -d '[:space:]' < "$ref_file")"
        path="$repo_dir/snapshots/$commit/$filename"
        if [ -e "$path" ]; then
            printf "%s" "$path"
            return 0
        fi
    fi

    find "$repo_dir/snapshots" \( -type f -o -type l \) 2>/dev/null |
        awk -v name="/$filename" '$0 ~ name "$" { print; found=1; exit } END { exit found ? 0 : 1 }'
}

find_llama_cli() {
    if [ -n "${LLAMA_CLI_BIN:-}" ] && [ -x "$LLAMA_CLI_BIN" ]; then
        printf "%s" "$LLAMA_CLI_BIN"
        return 0
    fi

    if command -v llama-cli >/dev/null 2>&1; then
        command -v llama-cli
        return 0
    fi

    find "$BUILD_DIR" -type f -name 'llama-cli' -perm -111 -print -quit 2>/dev/null |
        awk 'NF { print; found=1 } END { exit found ? 0 : 1 }'
}

disk_available_mb() {
    local path
    path="$(hf_cache_dir)"
    mkdir -p "$path"
    df -Pk "$path" | awk 'NR == 2 { print int($4 / 1024) }'
}

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
CPU_CORES="$(cpu_cores)"
DISK_FREE_MB="$(disk_available_mb)"
ACCELERATOR="cpu"
GPU_NAME=""
GPU_TOTAL_MB=0
GPU_FREE_MB=0

if [ "$OS_NAME" = "Darwin" ]; then
    TOTAL_RAM_MB="$(mac_total_mem_mb)"
    AVAILABLE_RAM_MB="$(mac_available_mem_mb)"
    if [ "$ARCH_NAME" = "arm64" ]; then
        ACCELERATOR="apple-unified"
        GPU_NAME="$(detect_apple_gpu_name || true)"
        GPU_NAME="${GPU_NAME:-Apple Silicon}"
        GPU_TOTAL_MB="$TOTAL_RAM_MB"
        GPU_FREE_MB="$AVAILABLE_RAM_MB"
    fi
else
    TOTAL_RAM_MB="$(linux_mem_mb MemTotal)"
    AVAILABLE_RAM_MB="$(linux_mem_mb MemAvailable)"
    NVIDIA_INFO="$(detect_nvidia_gpu || true)"
    if [ -n "$NVIDIA_INFO" ]; then
        ACCELERATOR="nvidia"
        GPU_NAME="$(trim "${NVIDIA_INFO%%|*}")"
        NVIDIA_REST="${NVIDIA_INFO#*|}"
        GPU_TOTAL_MB="${NVIDIA_REST%%|*}"
        GPU_FREE_MB="${NVIDIA_REST##*|}"
    fi
fi

if [ -z "${TOTAL_RAM_MB:-}" ] || [ "$TOTAL_RAM_MB" -eq 0 ]; then
    TOTAL_RAM_MB="$AVAILABLE_RAM_MB"
fi

RAM_SOFT_POOL_MB=$((TOTAL_RAM_MB * 75 / 100))
if [ "$AVAILABLE_RAM_MB" -gt "$RAM_SOFT_POOL_MB" ]; then
    RAM_SOFT_POOL_MB="$AVAILABLE_RAM_MB"
fi

SELECTED_ACCELERATOR="$ACCELERATOR"
MEMORY_POOL_MB="$RAM_SOFT_POOL_MB"
if [ "$ACCELERATOR" = "nvidia" ]; then
    MEMORY_POOL_MB="$GPU_FREE_MB"
elif [ "$ACCELERATOR" = "apple-unified" ]; then
    MEMORY_POOL_MB="$GPU_FREE_MB"
fi

OVERHEAD_MB="$(estimate_runtime_overhead_mb "$N_CTX")"
PREFERRED_QUANT="$(preferred_quant_for_hardware "$ACCELERATOR" "$GPU_NAME" "$GPU_FREE_MB" "$TOTAL_RAM_MB" "$CPU_CORES")"
PREFERRED_QUANT="$(apply_speed_target "$PREFERRED_QUANT" "$TARGET_TOKENS_PER_SEC")"

if ! SELECTED_QUANT="$(select_quant "$PREFERRED_QUANT" "$MEMORY_POOL_MB" "$DISK_FREE_MB" "$OVERHEAD_MB")"; then
    if [ "$ACCELERATOR" = "nvidia" ]; then
        SELECTED_ACCELERATOR="cpu"
        MEMORY_POOL_MB="$RAM_SOFT_POOL_MB"
        PREFERRED_QUANT="$(preferred_quant_for_hardware "cpu" "" 0 "$TOTAL_RAM_MB" "$CPU_CORES")"
        PREFERRED_QUANT="$(apply_speed_target "$PREFERRED_QUANT" "$TARGET_TOKENS_PER_SEC")"
        SELECTED_QUANT="$(select_quant "$PREFERRED_QUANT" "$MEMORY_POOL_MB" "$DISK_FREE_MB" "$OVERHEAD_MB" || true)"
    fi
fi

if [ -z "${SELECTED_QUANT:-}" ]; then
    echo "No Falcon 7B quantization fits this machine." >&2
    exit 1
fi

MODEL_NAME="$(quant_filename "$SELECTED_QUANT")"
MODEL_PATH="$(cached_model_path "$MODEL_REPO" "$MODEL_NAME" || true)"
LLAMA_CLI="$(find_llama_cli || true)"

if [ "$PRINT_PATH_ONLY" -ne 1 ]; then
    echo "Hardware analysis:"
    echo "  os/arch:        $OS_NAME/$ARCH_NAME"
    echo "  cpu cores:      $CPU_CORES"
    echo "  ram total/free: ${TOTAL_RAM_MB} MiB / ${AVAILABLE_RAM_MB} MiB"
    echo "  cache dir:      $(hf_cache_dir)"
    echo "  disk free:      ${DISK_FREE_MB} MiB"
    if [ "$ACCELERATOR" = "nvidia" ]; then
        echo "  gpu:            NVIDIA $GPU_NAME (${GPU_FREE_MB}/${GPU_TOTAL_MB} MiB free)"
    elif [ "$ACCELERATOR" = "apple-unified" ]; then
        echo "  gpu:            $GPU_NAME (Apple unified memory)"
    else
        echo "  gpu:            none detected for automatic offload"
    fi
    echo "Model decision:"
    echo "  target speed:   ~${TARGET_TOKENS_PER_SEC} tokens/sec"
    echo "  accelerator:    $SELECTED_ACCELERATOR"
    echo "  preferred:      $PREFERRED_QUANT"
    echo "  selected:       $SELECTED_QUANT ($MODEL_NAME)"
    if [ -n "$MODEL_PATH" ]; then
        echo "  status:         already cached"
    else
        echo "  status:         needs download via llama-cli -hf"
    fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
    if [ -n "$MODEL_PATH" ]; then
        echo "$MODEL_PATH"
    else
        printf "Dry run command:"
        if [ -n "$LLAMA_CLI" ]; then
            printf " %q" "$LLAMA_CLI"
        else
            printf " llama-cli"
        fi
        printf " -hf %q -hff %q -p '' -n 0 -c %q -b %q\n" "$MODEL_REPO:$SELECTED_QUANT" "$MODEL_NAME" "$N_CTX" "$N_BATCH"
    fi
    exit 0
fi

if [ -z "$MODEL_PATH" ]; then
    if [ -z "$LLAMA_CLI" ]; then
        echo "llama-cli not found. Run ./scripts/build.sh or set LLAMA_CLI_BIN=/path/to/llama-cli." >&2
        exit 1
    fi

    echo "Downloading with llama-cli -hf..." >&2
    download_timeout="${LLAMA_CLI_DOWNLOAD_TIMEOUT:-3600}"
    waited=0

    if [ -n "${HF_TOKEN:-}" ]; then
        HF_TOKEN="$HF_TOKEN" "$LLAMA_CLI" -hf "$MODEL_REPO:$SELECTED_QUANT" -hff "$MODEL_NAME" -p "" -n 0 -c "$N_CTX" -b "$N_BATCH" >&2 &
    else
        "$LLAMA_CLI" -hf "$MODEL_REPO:$SELECTED_QUANT" -hff "$MODEL_NAME" -p "" -n 0 -c "$N_CTX" -b "$N_BATCH" >&2 &
    fi
    llama_pid="$!"

    while kill -0 "$llama_pid" 2>/dev/null; do
        MODEL_PATH="$(cached_model_path "$MODEL_REPO" "$MODEL_NAME" || true)"
        if [ -n "$MODEL_PATH" ]; then
            kill "$llama_pid" 2>/dev/null || true
            wait "$llama_pid" 2>/dev/null || true
            break
        fi

        if [ "$waited" -ge "$download_timeout" ]; then
            kill "$llama_pid" 2>/dev/null || true
            wait "$llama_pid" 2>/dev/null || true
            echo "Timed out while llama-cli was downloading the model." >&2
            exit 1
        fi

        sleep 1
        waited=$((waited + 1))
    done

    if [ -z "$MODEL_PATH" ]; then
        wait "$llama_pid" || true
    fi

    MODEL_PATH="$(cached_model_path "$MODEL_REPO" "$MODEL_NAME" || true)"
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Download completed, but the GGUF was not found in the llama.cpp cache." >&2
    exit 1
fi

if [ "$PRINT_PATH_ONLY" -eq 1 ]; then
    printf "%s\n" "$MODEL_PATH"
else
    echo "Model path:"
    echo "  $MODEL_PATH"
fi
