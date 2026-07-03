#include "../include/model_manager.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

std::string getenv_or_empty(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "";
}

std::string repo_cache_folder(std::string repo) {
    std::string folder = "models--";
    for (const char ch : repo) {
        if (ch == '/') {
            folder += "--";
        } else {
            folder += ch;
        }
    }
    return folder;
}

std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string read_file_trimmed(const std::filesystem::path & path) {
    std::ifstream input(path);
    std::string   value;
    std::getline(input, value);
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) { return ch == '\r' || ch == '\n' || ch == ' ' || ch == '\t'; }), value.end());
    return value;
}

long long runtime_overhead_mb(const int n_ctx) {
    long long overhead = 1536;
    if (n_ctx > 2048) {
        overhead += ((n_ctx - 2048 + 2047) / 2048) * 512;
    }
    return overhead;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

int preferred_rank_for_hardware(const HardwareProfile & hardware) {
    const std::string gpu_name = lower_copy(hardware.gpu_name);

    if (hardware.accelerator == "nvidia") {
        if (hardware.gpu_free_mb >= 18000) {
            return 5;
        }
        if (hardware.gpu_free_mb >= 12000) {
            return 4;
        }
        if (hardware.gpu_free_mb >= 8500) {
            return 3;
        }
        if (hardware.gpu_free_mb >= 6000) {
            return 2;
        }
        return 1;
    }

    if (hardware.accelerator == "apple-unified") {
        if (gpu_name.find("ultra") != std::string::npos || gpu_name.find("max") != std::string::npos) {
            return hardware.total_ram_mb >= 64000 ? 4 : 3;
        }
        if (gpu_name.find("pro") != std::string::npos) {
            return hardware.total_ram_mb >= 32000 ? 3 : 2;
        }
        if (hardware.total_ram_mb >= 32000) {
            return 3;
        }
        if (hardware.total_ram_mb >= 16000) {
            return 2;
        }
        return 1;
    }

    if (hardware.cpu_cores >= 16 && hardware.total_ram_mb >= 32000) {
        return 2;
    }
    return 1;
}

int apply_speed_target(int rank, const int target_tps) {
    if (target_tps >= 45) {
        --rank;
    } else if (target_tps <= 20) {
        ++rank;
    }
    return std::clamp(rank, 1, 5);
}

bool model_fits(const ModelInfo & model, const HardwareProfile & hardware, const long long overhead_mb) {
    return hardware.memory_pool_mb >= model.size_mb + overhead_mb && hardware.disk_free_mb >= model.size_mb + 1024;
}

int select_model_index(const std::vector<ModelStatus> & models, int preferred_rank) {
    while (preferred_rank >= 1) {
        for (int index = 0; index < static_cast<int>(models.size()); ++index) {
            if (models[index].info.rank == preferred_rank && models[index].fits) {
                return index;
            }
        }
        --preferred_rank;
    }
    return 0;
}

bool is_executable(const std::filesystem::path & path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && access(path.c_str(), X_OK) == 0;
}

std::vector<std::filesystem::path> split_path_env(const std::string & value) {
    std::vector<std::filesystem::path> paths;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ':')) {
        if (!item.empty()) {
            paths.emplace_back(item);
        }
    }
    return paths;
}

}  // namespace

std::string hf_cache_dir() {
    if (!getenv_or_empty("LLAMA_CACHE").empty()) {
        return getenv_or_empty("LLAMA_CACHE");
    }
    if (!getenv_or_empty("HF_HUB_CACHE").empty()) {
        return getenv_or_empty("HF_HUB_CACHE");
    }
    if (!getenv_or_empty("HUGGINGFACE_HUB_CACHE").empty()) {
        return getenv_or_empty("HUGGINGFACE_HUB_CACHE");
    }
    if (!getenv_or_empty("HF_HOME").empty()) {
        return getenv_or_empty("HF_HOME") + "/hub";
    }
    if (!getenv_or_empty("XDG_CACHE_HOME").empty()) {
        return getenv_or_empty("XDG_CACHE_HOME") + "/huggingface/hub";
    }
    return getenv_or_empty("HOME") + "/.cache/huggingface/hub";
}

std::string cached_model_path(const std::string & repo, const std::string & filename) {
    const std::filesystem::path repo_dir = std::filesystem::path(hf_cache_dir()) / repo_cache_folder(repo);
    const std::filesystem::path ref_file = repo_dir / "refs" / "main";

    if (std::filesystem::exists(ref_file)) {
        const std::string commit = read_file_trimmed(ref_file);
        const auto        path = repo_dir / "snapshots" / commit / filename;
        if (std::filesystem::exists(path)) {
            return path.string();
        }
    }

    const auto snapshots = repo_dir / "snapshots";
    if (!std::filesystem::exists(snapshots)) {
        return "";
    }

    std::error_code error;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(snapshots, error)) {
        if (entry.path().filename() == filename) {
            return entry.path().string();
        }
    }
    return "";
}

std::string find_llama_cli(const AppConfig & config) {
    if (!config.llama_cli.empty() && is_executable(config.llama_cli)) {
        return config.llama_cli;
    }

    const std::string env_cli = getenv_or_empty("LLAMA_CLI_BIN");
    if (!env_cli.empty() && is_executable(env_cli)) {
        return env_cli;
    }

    for (const auto & path : split_path_env(getenv_or_empty("PATH"))) {
        const auto candidate = path / "llama-cli";
        if (is_executable(candidate)) {
            return candidate.string();
        }
    }

    if (!config.build_dir.empty() && std::filesystem::exists(config.build_dir)) {
        std::error_code error;
        for (const auto & entry : std::filesystem::recursive_directory_iterator(config.build_dir, error)) {
            if (entry.path().filename() == "llama-cli" && is_executable(entry.path())) {
                return entry.path().string();
            }
        }
    }

    return "";
}

ModelDecision build_model_decision(const AppConfig & config) {
    ModelDecision decision;
    decision.hardware = detect_hardware_profile(hf_cache_dir());

    const long long overhead_mb = runtime_overhead_mb(config.n_ctx);
    for (const auto & model : model_catalog()) {
        ModelStatus status;
        status.info = model;
        status.path = cached_model_path(config.model_repo, model.filename);
        status.cached = !status.path.empty();
        status.fits = model_fits(model, decision.hardware, overhead_mb);
        decision.models.push_back(status);
    }

    int preferred_rank = preferred_rank_for_hardware(decision.hardware);
    preferred_rank = apply_speed_target(preferred_rank, config.target_tokens_per_sec);
    decision.recommended_index = select_model_index(decision.models, preferred_rank);
    decision.models[decision.recommended_index].recommended = true;
    decision.selected_accelerator = decision.hardware.accelerator;

    if (!decision.models[decision.recommended_index].fits) {
        decision.reason = "No ideal quantization fits comfortably; using the smallest available option.";
    } else if (decision.hardware.accelerator == "cpu") {
        decision.reason = "CPU-only profile: selecting the smallest practical quantization; 30 tokens/sec may be unlikely.";
    } else {
        decision.reason = "Recommended from detected memory, disk space, accelerator, and target speed.";
    }

    return decision;
}

bool download_model_with_llama_cli(const AppConfig & config, const ModelInfo & model, std::string & output_path, std::string & error) {
    const std::string llama_cli = find_llama_cli(config);
    if (llama_cli.empty()) {
        error = "llama-cli not found. Build first with ./scripts/build.sh or set LLAMA_CLI_BIN.";
        return false;
    }

    const std::string command = shell_quote(llama_cli) + " -hf " + shell_quote(config.model_repo + ":" + model.quant) + " -hff " +
                                shell_quote(model.filename) + " -p '' -n 0 -c " + std::to_string(config.n_ctx) + " -b " +
                                std::to_string(config.n_batch) + " 2>&1";

    std::array<char, 512> buffer{};
    FILE *               pipe = popen(command.c_str(), "r");
    if (!pipe) {
        error = "Failed to start llama-cli.";
        return false;
    }

    std::string last_output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        last_output = buffer.data();
    }

    const int exit_code = pclose(pipe);
    output_path = cached_model_path(config.model_repo, model.filename);
    if (!output_path.empty()) {
        return true;
    }

    error = exit_code == 0 ? "Download finished, but the GGUF was not found in the cache." : "llama-cli failed while downloading the model.";
    if (!last_output.empty()) {
        error += " Last output: " + last_output;
    }
    return false;
}
