#include "../include/model_manager.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <ctime>
#include <unordered_set>

namespace {

std::string getenv_or_empty(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "";
}

struct QuantSpec {
    std::string label;
    int         rank = 0;
};

struct LastUsedModel {
    std::string path;
};

std::string normalize_token_text(std::string value, const char replacement) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    for (char & ch : value) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = replacement;
        }
    }
    return value;
}

std::string normalize_quant_text(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    for (char & ch : value) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return value;
}

bool looks_like_llama3_8b(const std::string & value) {
    const std::string text = normalize_token_text(value, '-');
    return text.find("llama-3-8b") != std::string::npos || text.find("llama3-8b") != std::string::npos;
}

bool looks_like_instruct(const std::string & value) {
    return normalize_token_text(value, '-').find("instruct") != std::string::npos;
}

std::optional<QuantSpec> quant_spec_from_name(const std::string & value) {
    const std::string text = normalize_quant_text(value);

    const std::array<QuantSpec, 15> specs = {
        QuantSpec{"IQ4_XS", 9},
        QuantSpec{"IQ4_NL", 10},
        QuantSpec{"Q4_K_S", 10},
        QuantSpec{"Q4_0", 10},
        QuantSpec{"Q4_K_M", 11},
        QuantSpec{"Q4_1", 11},
        QuantSpec{"Q5_K_S", 12},
        QuantSpec{"Q5_0", 12},
        QuantSpec{"Q5_K_M", 13},
        QuantSpec{"Q5_1", 13},
        QuantSpec{"Q6_K", 14},
        QuantSpec{"Q8_0", 15},
        QuantSpec{"FP16", 16},
        QuantSpec{"F_16", 16},
        QuantSpec{"FP32", 17},
    };

    for (const auto & spec : specs) {
        if (text.find(spec.label) != std::string::npos) {
            QuantSpec normalized = spec;
            if (normalized.label == "F_16") {
                normalized.label = "FP16";
            }
            return normalized;
        }
    }

    if (text.find("F32") != std::string::npos) {
        return QuantSpec{"FP32", 17};
    }
    if (text.find("F16") != std::string::npos) {
        return QuantSpec{"FP16", 16};
    }

    return std::nullopt;
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

std::string json_escape(const std::string & value) {
    std::string escaped;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

std::optional<std::string> json_string_field(const std::string & content, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    const auto        key_pos = content.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }

    const auto colon_pos = content.find(':', key_pos + needle.size());
    if (colon_pos == std::string::npos) {
        return std::nullopt;
    }

    auto pos = colon_pos + 1;
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos >= content.size() || content[pos] != '"') {
        return std::nullopt;
    }
    ++pos;

    std::string value;
    bool        escaped = false;
    for (; pos < content.size(); ++pos) {
        const char ch = content[pos];
        if (escaped) {
            switch (ch) {
                case 'n':
                    value += '\n';
                    break;
                case 'r':
                    value += '\r';
                    break;
                case 't':
                    value += '\t';
                    break;
                default:
                    value += ch;
                    break;
            }
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else if (ch == '"') {
            return value;
        } else {
            value += ch;
        }
    }

    return std::nullopt;
}

std::filesystem::path detect_llama_config_path() {
    const std::string explicit_config = getenv_or_empty("DETECT_LLAMA_CONFIG");
    if (!explicit_config.empty()) {
        return explicit_config;
    }

    const std::string xdg_config = getenv_or_empty("XDG_CONFIG_HOME");
    const std::string home = getenv_or_empty("HOME");
    const auto        base = !xdg_config.empty() ? std::filesystem::path(xdg_config)
                                                 : std::filesystem::path(home) / ".config";
    return base / "detectllama" / "config.json";
}

std::optional<LastUsedModel> load_last_used_model() {
    const auto config_path = detect_llama_config_path();
    std::ifstream input(config_path);
    if (!input) {
        return std::nullopt;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();

    const auto path = json_string_field(buffer.str(), "path");
    if (!path || path->empty()) {
        return std::nullopt;
    }

    return LastUsedModel{*path};
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

bool looks_below_four_bit_quant(const std::string & value) {
    std::string text = value;
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    const std::array<std::string, 15> patterns = {
        "iq1", "iq2", "iq3",
        "q2_", "q2-", ".q2", "_q2",
        "q3_", "q3-", ".q3", "_q3",
        "q2k", "q3k", "q2.k", "q3.k",
    };

    for (const auto & pattern : patterns) {
        if (text.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::filesystem::path default_llama_cache_dir() {
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

std::filesystem::path llama_cache_dir() {
    const std::string llama_cache = getenv_or_empty("LLAMA_CACHE");
    if (!llama_cache.empty()) {
        return llama_cache;
    }
    return default_llama_cache_dir();
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

long long runtime_overhead_mb(const HardwareProfile & hardware, const int n_ctx) {
    long long overhead = 1536;
    if (hardware.accelerator == "apple-unified") {
        overhead = 768;
    } else if (hardware.accelerator == "nvidia") {
        overhead = 1024;
    }

    if (n_ctx > 2048) {
        const long long extra_context_chunks = (n_ctx - 2048 + 2047) / 2048;
        overhead += extra_context_chunks * (hardware.accelerator == "apple-unified" ? 384 : 512);
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
        if (hardware.gpu_free_mb >= 24000) {
            return 16;
        }
        if (hardware.gpu_free_mb >= 18000) {
            return 15;
        }
        if (hardware.gpu_free_mb >= 12000) {
            return 14;
        }
        if (hardware.gpu_free_mb >= 8500) {
            return 13;
        }
        if (hardware.gpu_free_mb >= 6000) {
            return 11;
        }
        return 10;
    }

    if (hardware.accelerator == "apple-unified") {
        if (gpu_name.find("ultra") != std::string::npos || gpu_name.find("max") != std::string::npos) {
            return 16;
        }
        if (gpu_name.find("pro") != std::string::npos) {
            return hardware.total_ram_mb >= 32000 ? 16 : 14;
        }
        if (hardware.total_ram_mb >= 32000) {
            return 16;
        }
        if (hardware.total_ram_mb >= 16000) {
            return 15;
        }
        if (hardware.total_ram_mb >= 7000) {
            return 13;
        }
        return 10;
    }

    if (hardware.cpu_cores >= 16 && hardware.total_ram_mb >= 32000) {
        return 11;
    }
    return 10;
}

int apply_speed_target(int rank, const int target_tps) {
    if (target_tps >= 80) {
        rank -= 2;
    } else if (target_tps <= 20) {
        ++rank;
    }
    // FP32 stays available for manual selection, but automatic recommendations top out at FP16.
    return std::clamp(rank, 1, 16);
}

bool model_fits(const ModelInfo & model, const HardwareProfile & hardware, const long long overhead_mb) {
    return hardware.memory_pool_mb >= model.size_mb + overhead_mb && hardware.disk_free_mb >= model.size_mb + 1024;
}

void append_gguf_paths_from_dir(const std::filesystem::path & directory, std::vector<std::filesystem::path> & paths) {
    if (directory.empty()) {
        return;
    }

    std::error_code error;
    if (!std::filesystem::exists(directory, error)) {
        return;
    }

    for (const auto & entry : std::filesystem::recursive_directory_iterator(directory, error)) {
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
}

std::vector<std::filesystem::path> split_model_dirs_env(const std::string & value) {
    std::vector<std::filesystem::path> dirs;
    std::string                        current;
#if defined(_WIN32)
    constexpr char separator = ';';
#else
    constexpr char separator = ':';
#endif

    for (const char ch : value) {
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

std::vector<std::filesystem::path> local_gguf_paths() {
    std::vector<std::filesystem::path> paths;
    std::vector<std::filesystem::path> dirs = {
        llama_cache_dir(),
        default_llama_cache_dir(),
        hf_cache_path(),
    };

    for (const auto & dir : split_model_dirs_env(getenv_or_empty("DETECT_LLAMA_MODEL_DIRS"))) {
        dirs.push_back(dir);
    }

    std::unordered_set<std::string> seen_dirs;
    for (const auto & dir : dirs) {
        if (dir.empty()) {
            continue;
        }
        const auto key = dir.lexically_normal().string();
        if (!seen_dirs.insert(key).second) {
            continue;
        }
        append_gguf_paths_from_dir(dir, paths);
    }

    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
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

bool describe_local_model_path_impl(const std::filesystem::path & path, ModelInfo & model) {
    if (!is_complete_gguf_path(path)) {
        return false;
    }

    const std::string filename = path.filename().string();
    if (!looks_like_llama3_8b(filename) || looks_below_four_bit_quant(filename)) {
        return false;
    }

    const auto quant = quant_spec_from_name(filename);
    if (!quant) {
        return false;
    }

    model = {};
    model.family = looks_like_instruct(filename) ? "Llama 3 8B Instruct" : "Llama 3 8B";
    model.quant = quant->label;
    model.repo.clear();
    model.filename = filename;

    std::error_code error;
    const auto      bytes = std::filesystem::file_size(path, error);
    if (!error) {
        model.size_mb = static_cast<int>((bytes + 1024 * 1024 - 1) / (1024 * 1024));
    }
    model.rank = quant->rank;
    return true;
}

std::string cached_model_path_from_paths(const std::vector<std::filesystem::path> & paths, const std::string & filename) {
    for (const auto & path : paths) {
        if (filename_matches_cached_gguf(path, filename)) {
            return path.string();
        }
    }
    return "";
}

bool is_base_model(const ModelStatus & model) {
    return lower_copy(model.info.family).find("instruct") == std::string::npos;
}

int select_matching_model_index(const std::vector<ModelStatus> & models,
                                const int                        preferred_rank,
                                const bool                       require_cached,
                                const bool                       require_catalog) {
    for (int rank = preferred_rank; rank >= 1; --rank) {
        for (const bool base_only : {true, false}) {
            for (int index = 0; index < static_cast<int>(models.size()); ++index) {
                const auto & model = models[index];
                if (model.info.rank != rank || !model.fits) {
                    continue;
                }
                if (require_cached && !model.cached) {
                    continue;
                }
                if (require_catalog && !model.catalog_model) {
                    continue;
                }
                if (base_only != is_base_model(model)) {
                    continue;
                }
                return index;
            }
        }
    }
    return -1;
}

int select_last_used_model_index(const std::vector<ModelStatus> & models) {
    for (int index = 0; index < static_cast<int>(models.size()); ++index) {
        const auto & model = models[index];
        if (model.last_used && model.cached && model.fits) {
            return index;
        }
    }
    return -1;
}

int select_model_index(const std::vector<ModelStatus> & models, const int preferred_rank) {
    const int last_used_index = select_last_used_model_index(models);
    if (last_used_index >= 0) {
        return last_used_index;
    }

    const int local_index = select_matching_model_index(models, preferred_rank, true, false);
    if (local_index >= 0) {
        return local_index;
    }

    const int catalog_index = select_matching_model_index(models, preferred_rank, false, true);
    if (catalog_index >= 0) {
        return catalog_index;
    }

    return 0;
}

}  // namespace

std::string hf_cache_dir() {
    return hf_cache_path().string();
}

std::string cached_model_path(const std::string & repo, const std::string & filename) {
    const auto local_paths = local_gguf_paths();
    const auto local_match = cached_model_path_from_paths(local_paths, filename);
    if (!local_match.empty()) {
        return local_match;
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

    const long long overhead_mb = runtime_overhead_mb(decision.hardware, config.n_ctx);
    auto            local_paths = local_gguf_paths();
    const auto      last_used = load_last_used_model();
    std::string     last_used_key;
    if (last_used) {
        std::error_code error;
        if (std::filesystem::exists(last_used->path, error)) {
            const auto path = std::filesystem::path(last_used->path);
            local_paths.push_back(path);
            last_used_key = path_key(path);
        }
    }

    for (const auto & model : model_catalog()) {
        ModelStatus status;
        status.info = model;
        status.path = cached_model_path_from_paths(local_paths, model.filename);
        status.cached = !status.path.empty();
        status.fits = model_fits(model, decision.hardware, overhead_mb);
        status.last_used = !last_used_key.empty() && !status.path.empty() && path_key(status.path) == last_used_key;
        decision.models.push_back(status);
    }

    std::unordered_set<std::string> known_paths;
    for (const auto & model : decision.models) {
        if (!model.path.empty()) {
            known_paths.insert(path_key(model.path));
        }
    }

    for (const auto & path : local_paths) {
        if (!known_paths.insert(path_key(path)).second) {
            continue;
        }

        ModelStatus status;
        if (!describe_local_model_path_impl(path, status.info)) {
            continue;
        }
        status.path = path.string();
        status.cached = true;
        status.fits = model_fits(status.info, decision.hardware, overhead_mb);
        status.downloadable = false;
        status.catalog_model = false;
        status.last_used = !last_used_key.empty() && path_key(path) == last_used_key;
        decision.models.push_back(status);
    }

    int preferred_rank = preferred_rank_for_hardware(decision.hardware);
    preferred_rank = apply_speed_target(preferred_rank, config.target_tokens_per_sec);
    decision.recommended_index = select_model_index(decision.models, preferred_rank);
    decision.models[decision.recommended_index].recommended = true;
    decision.selected_accelerator = decision.hardware.accelerator;

    const auto & selected = decision.models[decision.recommended_index];
    if (selected.last_used) {
        decision.reason = "Previous DetectLlama model found locally and selected.";
    } else if (selected.cached) {
        decision.reason = "Local Llama 3 8B GGUF found; selecting the best compatible local model.";
    } else if (!selected.fits) {
        decision.reason = "No ideal quantization fits comfortably; using the smallest available option.";
    } else if (!selected.cached && selected.downloadable) {
        decision.reason = "No compatible local Llama 3 8B GGUF found; selected a catalog model for manual download.";
    } else if (decision.hardware.accelerator == "cpu") {
        decision.reason = "CPU-only profile: selecting the smallest practical quantization; 30 tokens/sec may be unlikely.";
    } else if (decision.hardware.accelerator == "apple-unified") {
        decision.reason = "Apple unified memory profile: preferring higher-quality quantization while keeping a fit fallback.";
    } else {
        decision.reason = "Recommended from detected memory, disk space, accelerator, and target speed.";
    }

    return decision;
}

bool save_last_used_model(const ModelInfo & model, const std::string & path, std::string & error) {
    const auto config_path = detect_llama_config_path();

    std::error_code fs_error;
    std::filesystem::create_directories(config_path.parent_path(), fs_error);
    if (fs_error) {
        error = "Could not create DetectLlama config directory: " + fs_error.message();
        return false;
    }

    std::ofstream output(config_path);
    if (!output) {
        error = "Could not write DetectLlama config: " + config_path.string();
        return false;
    }

    output << "{\n"
           << "  \"last_model\": {\n"
           << "    \"path\": \"" << json_escape(path) << "\",\n"
           << "    \"label\": \"" << json_escape(model_label(model)) << "\",\n"
           << "    \"family\": \"" << json_escape(model.family) << "\",\n"
           << "    \"quant\": \"" << json_escape(model.quant) << "\",\n"
           << "    \"repo\": \"" << json_escape(model.repo) << "\",\n"
           << "    \"filename\": \"" << json_escape(model.filename) << "\",\n"
           << "    \"last_loaded_at\": " << static_cast<long long>(std::time(nullptr)) << "\n"
           << "  }\n"
           << "}\n";

    if (!output) {
        error = "Could not finish writing DetectLlama config: " + config_path.string();
        return false;
    }

    return true;
}

bool describe_local_model_path(const std::string & path, ModelInfo & model) {
    return describe_local_model_path_impl(std::filesystem::path(path), model);
}

bool download_public_model(const ModelInfo & model, std::string & output_path, std::string & error) {
    clear_hf_token_environment();

    const std::filesystem::path final_path = public_model_cache_path(model.repo, model);
    const std::filesystem::path partial_path = final_path.string() + ".part";

    output_path = cached_model_path(model.repo, model.filename);
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
                                shell_quote(partial_path.string()) + " " + shell_quote(public_model_url(model.repo, model.filename)) + " 2>&1";

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

    output_path = cached_model_path(model.repo, model.filename);
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
