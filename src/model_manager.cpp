#include "../include/model_manager.h"
#include "../include/app_config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string getenv_or_empty(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "";
}

std::filesystem::path default_llama_cache_dir() {
    const std::string home      = getenv_or_empty("HOME");
    const std::string xdg_cache = getenv_or_empty("XDG_CACHE_HOME");
#if defined(__APPLE__)
    return std::filesystem::path(home) / "Library" / "Caches" / "llama.cpp";
#else
    return xdg_cache.empty() ? std::filesystem::path(home) / ".cache" / "llama.cpp" :
                               std::filesystem::path(xdg_cache) / "llama.cpp";
#endif
}

std::filesystem::path llama_cache_dir() {
    const std::string configured = getenv_or_empty("LLAMA_CACHE");
    return configured.empty() ? default_llama_cache_dir() : std::filesystem::path(configured);
}

std::filesystem::path hf_cache_path() {
    if (!getenv_or_empty("HF_HUB_CACHE").empty()) {
        return getenv_or_empty("HF_HUB_CACHE");
    }
    if (!getenv_or_empty("HUGGINGFACE_HUB_CACHE").empty()) {
        return getenv_or_empty("HUGGINGFACE_HUB_CACHE");
    }
    if (!getenv_or_empty("HF_HOME").empty()) {
        return std::filesystem::path(getenv_or_empty("HF_HOME")) / "hub";
    }
    if (!getenv_or_empty("XDG_CACHE_HOME").empty()) {
        return std::filesystem::path(getenv_or_empty("XDG_CACHE_HOME")) / "huggingface" / "hub";
    }
    return std::filesystem::path(getenv_or_empty("HOME")) / ".cache" / "huggingface" / "hub";
}

std::string llama_cache_repo_key(std::string repo) {
    std::replace(repo.begin(), repo.end(), '/', '_');
    return repo;
}

std::filesystem::path download_path() {
    const auto & model = detectllama_model();
    return llama_cache_dir() / (llama_cache_repo_key(model.repo) + ":" + model.quant + "_" + model.filename);
}

bool filename_matches(const std::filesystem::path & path) {
    const std::string name   = path.filename().string();
    const std::string suffix = "_" + detectllama_model().filename;
    return name == detectllama_model().filename ||
           (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0);
}

std::vector<std::filesystem::path> extra_model_dirs() {
    std::vector<std::filesystem::path> dirs;
    std::string current;
#if defined(_WIN32)
    constexpr char separator = ';';
#else
    constexpr char separator = ':';
#endif
    for (const char ch : getenv_or_empty("DETECT_LLAMA_MODEL_DIRS")) {
        if (ch == separator) {
            if (!current.empty()) {
                dirs.emplace_back(current);
                current.clear();
            }
        } else {
            current += ch;
        }
    }
    if (!current.empty()) {
        dirs.emplace_back(current);
    }
    return dirs;
}

std::string find_in_directory(const std::filesystem::path & directory) {
    std::error_code error;
    if (directory.empty() || !std::filesystem::exists(directory, error)) {
        return "";
    }
    for (const auto & entry : std::filesystem::recursive_directory_iterator(directory, error)) {
        if (error) {
            break;
        }
        std::error_code status_error;
        if (filename_matches(entry.path()) &&
            (std::filesystem::is_regular_file(entry.path(), status_error) ||
             std::filesystem::is_symlink(entry.path(), status_error))) {
            return entry.path().string();
        }
    }
    return "";
}

std::string url_encode(const std::string & value) {
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
        quoted += ch == '\'' ? "'\\''" : std::string(1, ch);
    }
    return quoted + "'";
}

void clear_hf_token_environment() {
    const std::array<const char *, 4> names = {
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"
    };
    for (const char * name : names) {
#if defined(_WIN32)
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }
}

}  // namespace

const ModelInfo & detectllama_model() {
    static const ModelInfo model = {
        "Llama 3 8B",
        "Q4_0",
        "QuantFactory/Meta-Llama-3-8B-GGUF",
        "Meta-Llama-3-8B.Q4_0.gguf",
        4662,
        kDetectionThreshold,
        kContextLength,
        kCalibrationId,
    };
    return model;
}

std::string model_label() {
    return detectllama_model().family + " " + detectllama_model().quant;
}

std::string hf_cache_dir() {
    return hf_cache_path().string();
}

std::string cached_model_path() {
    std::vector<std::filesystem::path> dirs = {
        llama_cache_dir(),
        default_llama_cache_dir(),
        hf_cache_path(),
    };
    const auto extra = extra_model_dirs();
    dirs.insert(dirs.end(), extra.begin(), extra.end());

    std::unordered_set<std::string> seen;
    for (const auto & dir : dirs) {
        const std::string key = dir.lexically_normal().string();
        if (!key.empty() && seen.insert(key).second) {
            const std::string path = find_in_directory(dir);
            if (!path.empty()) {
                return path;
            }
        }
    }
    return "";
}

ModelState inspect_model_state() {
    ModelState state;
    state.hardware = detect_hardware_profile(llama_cache_dir().string());
    state.path     = cached_model_path();
    state.cached   = !state.path.empty();
    state.fits     = state.hardware.memory_pool_mb >= detectllama_model().size_mb + 1536 &&
                 state.hardware.disk_free_mb >= detectllama_model().size_mb + 1024;
    return state;
}

bool is_detectllama_model_path(const std::string & path) {
    std::error_code error;
    const auto      file = std::filesystem::path(path);
    return filename_matches(file) && std::filesystem::exists(file, error) &&
           std::filesystem::is_regular_file(file, error);
}

bool download_model(std::string & output_path, std::string & error) {
    namespace fs = std::filesystem;
    clear_hf_token_environment();

    const auto target = download_path();
    const auto partial = fs::path(target.string() + ".part");
    std::error_code fs_error;
    fs::create_directories(target.parent_path(), fs_error);
    if (fs_error) {
        error = "Could not create model cache directory: " + fs_error.message();
        return false;
    }

    const auto & model = detectllama_model();
    const std::string url = "https://huggingface.co/" + url_encode(model.repo) + "/resolve/main/" +
                            url_encode(model.filename);
    const std::string command = "curl --fail --location --progress-bar --output " + shell_quote(partial.string()) +
                                " " + shell_quote(url);
    if (std::system(command.c_str()) != 0) {
        error = "Failed to download " + model_label() + ".";
        return false;
    }

    fs::rename(partial, target, fs_error);
    if (fs_error) {
        error = "Downloaded the model but could not finalize the cache file: " + fs_error.message();
        return false;
    }
    output_path = target.string();
    return true;
}
