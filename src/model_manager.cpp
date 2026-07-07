#include "../include/model_manager.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>

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

std::string llama_cache_repo_key(std::string repo) {
    std::replace(repo.begin(), repo.end(), '/', '_');
    return repo;
}

std::string url_encode_hf_path(const std::string & value) {
    std::ostringstream encoded;
    encoded << std::uppercase << std::hex;
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~' || ch == '/') {
            encoded << static_cast<char>(ch);
        } else {
            encoded << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
            encoded << std::setfill(' ');
        }
    }
    return encoded.str();
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

bool filename_matches_cached_gguf(const std::filesystem::path & path, const std::string & filename) {
    const std::string name = path.filename().string();
    const std::string llama_cache_suffix = "_" + filename;
    return name == filename || (name.size() > llama_cache_suffix.size() &&
                                name.compare(name.size() - llama_cache_suffix.size(), llama_cache_suffix.size(), llama_cache_suffix) == 0);
}

bool is_complete_gguf_path(const std::filesystem::path & path) {
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0;
}

std::filesystem::path llama_cache_dir() {
    const std::string llama_cache = getenv_or_empty("LLAMA_CACHE");
    if (!llama_cache.empty()) {
        return llama_cache;
    }

    const std::string home = getenv_or_empty("HOME");
    const std::string xdg_cache = getenv_or_empty("XDG_CACHE_HOME");
#if defined(__APPLE__)
    return std::filesystem::path(home) / "Library" / "Caches" / "llama.cpp";
#else
    if (!xdg_cache.empty()) {
        return std::filesystem::path(xdg_cache) / "llama.cpp";
    }
    return std::filesystem::path(home) / ".cache" / "llama.cpp";
#endif
}

std::filesystem::path public_model_cache_path(const std::string & repo, const ModelInfo & model) {
    return llama_cache_dir() / (llama_cache_repo_key(repo) + ":" + model.quant + "_" + model.filename);
}

std::string public_model_url(const std::string & repo, const std::string & filename) {
    return "https://huggingface.co/" + url_encode_hf_path(repo) + "/resolve/main/" + url_encode_hf_path(filename);
}

void clear_hf_token_environment() {
    const std::array<const char *, 4> token_env_names = {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_HUB_TOKEN",
    };

    for (const char * name : token_env_names) {
#if defined(_WIN32)
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }
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

std::vector<std::filesystem::path> llama_cached_gguf_paths() {
    std::vector<std::filesystem::path> paths;
    const auto                         cache_dir = llama_cache_dir();
    std::error_code                    error;

    if (!std::filesystem::exists(cache_dir, error)) {
        return paths;
    }

    for (const auto & entry : std::filesystem::recursive_directory_iterator(cache_dir, error)) {
        if (error) {
            break;
        }
        if (!is_complete_gguf_path(entry.path())) {
            continue;
        }

        std::error_code status_error;
        if (std::filesystem::is_regular_file(entry.path(), status_error) || std::filesystem::is_symlink(entry.path(), status_error)) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string path_key(const std::filesystem::path & path) {
    std::error_code error;
    const auto      canonical = std::filesystem::weakly_canonical(path, error);
    if (!error) {
        return canonical.string();
    }
    return std::filesystem::absolute(path, error).string();
}

ModelInfo local_model_info(const std::filesystem::path & path, int rank) {
    ModelInfo model;
    model.filename = path.filename().string();
    model.quant = path.stem().string();
    model.rank = rank;

    std::error_code error;
    const auto      bytes = std::filesystem::file_size(path, error);
    if (!error) {
        model.size_mb = static_cast<int>((bytes + 1024 * 1024 - 1) / (1024 * 1024));
    }

    return model;
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

}  // namespace

std::string hf_cache_dir() {
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
    for (const auto & path : llama_cached_gguf_paths()) {
        if (filename_matches_cached_gguf(path, filename)) {
            return path.string();
        }
    }

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
    std::error_code error;
    if (std::filesystem::exists(snapshots)) {
        for (const auto & entry : std::filesystem::recursive_directory_iterator(snapshots, error)) {
            if (entry.path().filename() == filename) {
                return entry.path().string();
            }
        }
    }

    return "";
}

ModelDecision build_model_decision(const AppConfig & config) {
    ModelDecision decision;
    decision.hardware = detect_hardware_profile(llama_cache_dir().string());

    const long long overhead_mb = runtime_overhead_mb(config.n_ctx);
    for (const auto & model : model_catalog()) {
        ModelStatus status;
        status.info = model;
        status.path = cached_model_path(config.model_repo, model.filename);
        status.cached = !status.path.empty();
        status.fits = model_fits(model, decision.hardware, overhead_mb);
        decision.models.push_back(status);
    }

    std::unordered_set<std::string> known_paths;
    for (const auto & model : decision.models) {
        if (!model.path.empty()) {
            known_paths.insert(path_key(model.path));
        }
    }

    int local_rank = 100;
    for (const auto & path : llama_cached_gguf_paths()) {
        if (!known_paths.insert(path_key(path)).second) {
            continue;
        }

        ModelStatus status;
        status.info = local_model_info(path, local_rank++);
        status.path = path.string();
        status.cached = true;
        status.fits = model_fits(status.info, decision.hardware, overhead_mb);
        status.downloadable = false;
        status.catalog_model = false;
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

bool download_public_model(const AppConfig & config, const ModelInfo & model, std::string & output_path, std::string & error) {
    clear_hf_token_environment();

    const std::filesystem::path final_path = public_model_cache_path(config.model_repo, model);
    const std::filesystem::path partial_path = final_path.string() + ".part";

    output_path = cached_model_path(config.model_repo, model.filename);
    if (!output_path.empty()) {
        return true;
    }

    std::error_code fs_error;
    std::filesystem::create_directories(final_path.parent_path(), fs_error);
    if (fs_error) {
        error = "DetectLlama could not create the llama.cpp cache directory: " + fs_error.message();
        return false;
    }

    const std::string command = "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HF_HUB_TOKEN curl "
                                "--fail --location --silent --show-error --retry 3 --continue-at - --output " +
                                shell_quote(partial_path.string()) + " " + shell_quote(public_model_url(config.model_repo, model.filename)) + " 2>&1";

    std::array<char, 512> buffer{};
    FILE *               pipe = popen(command.c_str(), "r");
    if (!pipe) {
        error = "Failed to start curl for anonymous public model download.";
        return false;
    }

    std::string last_output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        last_output = buffer.data();
    }

    const int exit_code = pclose(pipe);
    if (exit_code == 0 && std::filesystem::exists(partial_path)) {
        std::error_code fs_error;
        std::filesystem::rename(partial_path, final_path, fs_error);
        if (fs_error) {
            error = "Download finished, but DetectLlama could not finalize the GGUF file in the llama.cpp cache: " + fs_error.message();
            return false;
        }
    }

    output_path = cached_model_path(config.model_repo, model.filename);
    if (!output_path.empty()) {
        return true;
    }

    error = exit_code == 0 ? "Download finished, but the GGUF was not found in the llama.cpp cache."
                           : "Anonymous public download failed.";
    if (!last_output.empty()) {
        error += " Last output: " + last_output;
    }
    return false;
}
