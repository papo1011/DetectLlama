#include "../include/backend.h"

#include "../include/io.h"
#include "../include/llama_state.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace {

bool is_hex(const char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F');
}

int hex_value(const char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return value - 'A' + 10;
}

std::string percent_decode(const std::string & value) {
    std::string decoded;
    decoded.reserve(value.size());

    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '%' && index + 2 < value.size() && is_hex(value[index + 1]) && is_hex(value[index + 2])) {
            decoded.push_back(static_cast<char>((hex_value(value[index + 1]) << 4) + hex_value(value[index + 2])));
            index += 2;
            continue;
        }
        decoded.push_back(value[index]);
    }

    return decoded;
}

std::string unescape_terminal_path(const std::string & value) {
    std::string unescaped;
    unescaped.reserve(value.size());

    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '\\' && index + 1 < value.size()) {
            const char next = value[index + 1];
            if (next == ' ' || next == '\\' || next == '\'' || next == '"' || next == '(' || next == ')') {
                unescaped.push_back(next);
                ++index;
                continue;
            }
        }
        unescaped.push_back(value[index]);
    }

    return unescaped;
}

std::string hardware_summary(const HardwareProfile & hardware) {
    std::string summary = hardware.os_name + "/" + hardware.arch_name + " | " + std::to_string(hardware.cpu_cores) +
                          " CPU cores | RAM " + std::to_string(hardware.available_ram_mb) + "/" +
                          std::to_string(hardware.total_ram_mb) + " MiB | " + hardware.accelerator;
    if (!hardware.gpu_name.empty()) {
        summary += " " + hardware.gpu_name;
    }
    return summary;
}

}  // namespace

std::string format_fixed(const double value, const int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

double ai_probability_from_score(const double score) {
    const double probability = 1.0 / (1.0 + std::exp(1.35 * score));
    return std::clamp(probability, 0.0, 1.0);
}

std::string format_percent(const double probability) {
    return format_fixed(probability * 100.0, 1) + "%";
}

std::string interpret_score(const double score) {
    if (score >= 1.0) {
        return "Likely human-written";
    }
    if (score <= -1.0) {
        return "Likely AI-generated or model-like";
    }
    return "Ambiguous; compare against a calibrated threshold";
}

std::string trim_copy(const std::string & value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }

    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool starts_with_slash_command(const std::string & value, const std::string_view command) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.size() < command.size()) {
        return false;
    }

    const std::string head = lower_copy(trimmed.substr(0, command.size()));
    if (head != command) {
        return false;
    }

    return trimmed.size() == command.size() || std::isspace(static_cast<unsigned char>(trimmed[command.size()]));
}

std::string slash_command_argument(const std::string & value, const std::string_view command) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.size() <= command.size()) {
        return "";
    }
    return trim_copy(trimmed.substr(command.size()));
}

std::vector<std::string> slash_command_matches(const std::string & value) {
    static const std::vector<std::string> commands = { "/models", "/path" };
    const std::string                     trimmed = lower_copy(trim_copy(value));
    if (trimmed.empty() || trimmed.front() != '/' || trimmed.find_first_of(" \t\r\n") != std::string::npos) {
        return {};
    }

    std::vector<std::string> matches;
    for (const auto & command : commands) {
        if (command.rfind(trimmed, 0) == 0) {
            matches.push_back(command);
        }
    }
    return matches;
}

std::string command_description(const std::string & command) {
    if (command == "/models") {
        return "select or download a model";
    }
    if (command == "/path") {
        return "analyze a local .txt or .md file";
    }
    return "";
}

std::string normalize_dropped_path(const std::string & raw_path) {
    std::string path = trim_copy(raw_path);
    const auto  first_line_end = path.find_first_of("\r\n");
    if (first_line_end != std::string::npos) {
        path = trim_copy(path.substr(0, first_line_end));
    }

    if (path.size() >= 2 && ((path.front() == '"' && path.back() == '"') || (path.front() == '\'' && path.back() == '\''))) {
        path = path.substr(1, path.size() - 2);
    }

    constexpr std::string_view file_scheme = "file://";
    if (path.rfind(file_scheme, 0) == 0) {
        path = path.substr(file_scheme.size());
        constexpr std::string_view localhost = "localhost";
        if (path.rfind(localhost, 0) == 0) {
            path = path.substr(localhost.size());
        }
        path = percent_decode(path);
    }

    return trim_copy(unescape_terminal_path(path));
}

bool is_supported_input_file(const std::filesystem::path & path) {
    const std::string extension = lower_copy(path.extension().string());
    return extension == ".md" || extension == ".txt";
}

PromptParseResult parse_prompt_input(const std::string & raw_prompt) {
    namespace fs = std::filesystem;

    PromptParseResult result;
    const std::string trimmed_prompt = trim_copy(raw_prompt);
    if (trimmed_prompt.empty()) {
        result.action = PromptAction::Empty;
        result.message = "Write /models, /path <file>, or paste text before analyzing.";
        return result;
    }

    if (starts_with_slash_command(trimmed_prompt, "/models")) {
        const std::string arg = lower_copy(slash_command_argument(trimmed_prompt, "/models"));
        result.action = arg.empty() ? PromptAction::ModelPicker : PromptAction::ModelQuery;
        result.model_query = arg;
        return result;
    }

    if (trimmed_prompt.front() == '/' && !starts_with_slash_command(trimmed_prompt, "/path")) {
        result.action = PromptAction::UnknownCommand;
        result.message = "Unknown command. Available commands: /models and /path <file>.";
        return result;
    }

    if (starts_with_slash_command(trimmed_prompt, "/path")) {
        const std::string path = normalize_dropped_path(slash_command_argument(trimmed_prompt, "/path"));
        if (path.empty()) {
            result.action = PromptAction::UnknownCommand;
            result.message = "Use /path followed by a .md or .txt file path.";
            return result;
        }

        result.action = PromptAction::Analyze;
        result.input.kind = DetectionInputKind::File;
        result.input.value = path;
        result.input.source_label = "File: " + fs::path(path).filename().string();
        return result;
    }

    const std::string possible_path = normalize_dropped_path(trimmed_prompt);
    std::error_code   path_error;
    if (possible_path.find_first_of("\r\n") == std::string::npos && possible_path.size() < 4096 &&
        fs::exists(possible_path, path_error) && fs::is_regular_file(possible_path, path_error)) {
        result.action = PromptAction::Analyze;
        result.input.kind = DetectionInputKind::File;
        result.input.value = possible_path;
        result.input.source_label = "File: " + fs::path(possible_path).filename().string();
        return result;
    }

    result.action = PromptAction::Analyze;
    result.input.kind = DetectionInputKind::Text;
    result.input.value = trimmed_prompt;
    result.input.source_label = "Pasted text";
    return result;
}

void BackendSession::LlamaStateDeleter::operator()(LlamaState * state) const {
    if (state) {
        free_llama_state(*state);
        delete state;
    }
}

BackendSession::BackendSession(AppConfig config)
    : config_(std::move(config)), llama_(LlamaStatePtr(new LlamaState{}, LlamaStateDeleter{})) {}

BackendSession::~BackendSession() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    LlamaStatePtr old_llama;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        old_llama = std::move(llama_);
        snapshot_.model_ready = false;
    }
    old_llama.reset();
    if (backend_initialized_) {
        llama_backend_free();
        backend_initialized_ = false;
    }
}

BackendSnapshot BackendSession::snapshot() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return snapshot_;
}

void BackendSession::set_operation_status(const std::string & status) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    snapshot_.operation_status = status;
}

void BackendSession::clear_analysis() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    reset_analysis_fields_locked();
}

void BackendSession::reset_analysis_fields_locked() {
    snapshot_.operation_status = snapshot_.model_ready ? "Type / for commands, /path <file>, or paste text directly into the prompt."
                                                       : "Waiting for model.";
    snapshot_.score_text = "-";
    snapshot_.ai_probability = "-";
    snapshot_.input_source = "-";
    snapshot_.interpretation = snapshot_.model_ready ? "Ready to analyze files or pasted text." : "Waiting for model.";
    snapshot_.token_count = "-";
    snapshot_.elapsed = "-";
    snapshot_.speed = "-";
}

void BackendSession::ensure_backend_initialized() {
    if (!backend_initialized_) {
        llama_backend_init();
        backend_initialized_ = true;
    }
}

void BackendSession::initialize() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_status = "Profiling this machine and checking the llama.cpp cache...";
        snapshot_.operation_status = "Type / for commands, /path <file>, or paste text to detect.";
        snapshot_.interpretation = "Waiting for model.";
    }

    ModelDecision next_decision = build_model_decision(config_);
    ModelStatus   model_to_load;
    bool          should_load = false;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.decision = std::move(next_decision);
        snapshot_.decision_ready = true;
        snapshot_.selected_model_index = snapshot_.decision.recommended_index;
        snapshot_.profile_summary = hardware_summary(snapshot_.decision.hardware);

        const auto & recommended = snapshot_.decision.models[snapshot_.decision.recommended_index];
        if (recommended.cached) {
            model_to_load = recommended;
            should_load = true;
        } else {
            snapshot_.model_status = "Recommended model is not installed: " + model_label(recommended.info);
            snapshot_.operation_status = "Use /models to select and download a quantization.";
            snapshot_.interpretation = snapshot_.decision.reason;
        }
    }

    if (should_load) {
        load_model_status_unlocked(model_to_load);
    }
}

void BackendSession::refresh_model_cache_state_locked() {
    namespace fs = std::filesystem;

    for (auto & model : snapshot_.decision.models) {
        if (model.catalog_model) {
            model.path = cached_model_path(model.info.repo, model.info.filename);
            model.cached = !model.path.empty();
        } else {
            std::error_code error;
            model.cached = !model.path.empty() && fs::exists(model.path, error);
        }
    }
}

bool BackendSession::refresh_model_cache_state() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!snapshot_.decision_ready) {
        return false;
    }
    refresh_model_cache_state_locked();
    return true;
}

ModelStatus BackendSession::selected_model_locked() const {
    if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
        return {};
    }

    const int index = std::clamp(snapshot_.selected_model_index, 0, static_cast<int>(snapshot_.decision.models.size()) - 1);
    return snapshot_.decision.models[index];
}

bool BackendSession::select_model_index(const int index) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!snapshot_.decision_ready || snapshot_.decision.models.empty() || index < 0 ||
        index >= static_cast<int>(snapshot_.decision.models.size())) {
        return false;
    }
    snapshot_.selected_model_index = index;
    snapshot_.operation_status = "Selected " + model_label(snapshot_.decision.models[snapshot_.selected_model_index].info) + ".";
    return true;
}

void BackendSession::select_previous_model() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
        return;
    }
    snapshot_.selected_model_index = (snapshot_.selected_model_index + static_cast<int>(snapshot_.decision.models.size()) - 1) %
                                     static_cast<int>(snapshot_.decision.models.size());
    snapshot_.operation_status = "Selected " + model_label(snapshot_.decision.models[snapshot_.selected_model_index].info) + ".";
}

void BackendSession::select_next_model() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
        return;
    }
    snapshot_.selected_model_index = (snapshot_.selected_model_index + 1) % static_cast<int>(snapshot_.decision.models.size());
    snapshot_.operation_status = "Selected " + model_label(snapshot_.decision.models[snapshot_.selected_model_index].info) + ".";
}

bool BackendSession::load_model_status_unlocked(const ModelStatus & model) {
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_ready = false;
        snapshot_.loaded_model_quant.clear();
        snapshot_.loaded_model_path.clear();
        loaded_model_info_ = {};
        snapshot_.model_status = "Loading " + model_label(model.info) + " from local storage...";
        snapshot_.operation_status = "Model loading is running in the background.";
        snapshot_.interpretation = "Waiting for model.";
    }

    ensure_backend_initialized();

    auto       next = LlamaStatePtr(new LlamaState{}, LlamaStateDeleter{});
    const bool ok = setup_llama(*next, model.path, config_.use_gpu, config_.n_ctx, config_.n_batch);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (ok) {
            llama_ = next;
            snapshot_.model_ready = true;
            snapshot_.loaded_model_quant = model_label(model.info);
            snapshot_.loaded_model_path = model.path;
            loaded_model_info_ = model.info;
            snapshot_.model_status = "Model ready: " + model_label(model.info);
            snapshot_.operation_status = "Type / for commands, /path <file>, or paste text directly into the prompt.";
            snapshot_.interpretation = "Ready to analyze files or pasted text.";
        } else {
            snapshot_.model_status = "Failed to load model: " + model.path;
            snapshot_.operation_status = "Choose another cached model or download the recommended one.";
            snapshot_.interpretation = "No model is loaded.";
        }
    }
    return ok;
}

bool BackendSession::load_selected_model() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    ModelStatus model;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
            snapshot_.operation_status = "Model list is not ready yet.";
            return false;
        }
        refresh_model_cache_state_locked();
        model = selected_model_locked();
        if (!model.cached) {
            snapshot_.operation_status = "Selected model is not installed. Open /models and press Enter to download it.";
            return false;
        }
    }

    return load_model_status_unlocked(model);
}

bool BackendSession::download_selected_model_unlocked() {
    ModelStatus model;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
            snapshot_.operation_status = "Model list is not ready yet.";
            return false;
        }
        model = selected_model_locked();
        if (!model.downloadable) {
            snapshot_.operation_status = "This llama.cpp cache model is local-only and cannot be downloaded by DetectLlama.";
            return false;
        }
        snapshot_.model_ready = false;
        snapshot_.model_status = "Downloading " + model_label(model.info) + " anonymously from Hugging Face...";
        snapshot_.operation_status = "Download is running. The terminal stays inside DetectLlama.";
        snapshot_.interpretation = "Waiting for download.";
    }

    std::string output_path;
    std::string error;
    const bool  ok = download_public_model(model.info, output_path, error);
    ModelStatus downloaded = model;
    downloaded.path = output_path;
    downloaded.cached = ok;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        refresh_model_cache_state_locked();
        if (!ok) {
            snapshot_.model_status = error;
            snapshot_.operation_status = "Download failed. Check network access and that the model repo is public.";
            snapshot_.interpretation = "No model is loaded.";
        }
    }

    if (!ok) {
        return false;
    }
    return load_model_status_unlocked(downloaded);
}

bool BackendSession::activate_selected_model() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    ModelStatus model;
    bool        cached = false;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
            snapshot_.operation_status = "Model list is not ready yet.";
            return false;
        }
        refresh_model_cache_state_locked();
        model = selected_model_locked();
        cached = model.cached;
        if (!cached && !model.downloadable) {
            snapshot_.operation_status = "That llama.cpp cache model is no longer available on disk.";
            return false;
        }
        snapshot_.operation_status = cached ? "Loading " + model_label(model.info) + " from cache."
                                            : "Downloading " + model_label(model.info) + " anonymously.";
    }

    if (cached) {
        return load_model_status_unlocked(model);
    }
    return download_selected_model_unlocked();
}

bool BackendSession::load_model_by_query(const std::string & query, const bool download_if_missing) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    ModelStatus model;
    bool        should_download = false;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!snapshot_.decision_ready || snapshot_.decision.models.empty()) {
            snapshot_.operation_status = "Model list is not ready yet.";
            return false;
        }

        refresh_model_cache_state_locked();
        int next_index = -1;
        const std::string needle = lower_copy(query);
        for (int index = 0; index < static_cast<int>(snapshot_.decision.models.size()); ++index) {
            const auto & candidate = snapshot_.decision.models[index];
            if (lower_copy(model_label(candidate.info)).find(needle) != std::string::npos ||
                lower_copy(candidate.info.family).find(needle) != std::string::npos ||
                lower_copy(candidate.info.quant).find(needle) != std::string::npos ||
                lower_copy(candidate.info.repo).find(needle) != std::string::npos ||
                lower_copy(candidate.info.filename).find(needle) != std::string::npos ||
                std::to_string(candidate.info.rank) == needle) {
                next_index = index;
                break;
            }
        }
        if (next_index < 0) {
            snapshot_.operation_status = "Unknown model. Use /models to open the selector or /models <quant>.";
            return false;
        }

        snapshot_.selected_model_index = next_index;
        model = snapshot_.decision.models[snapshot_.selected_model_index];
        snapshot_.operation_status = "Selected " + model_label(model.info) + ".";
        if (model.cached) {
            snapshot_.operation_status += " Loading from cache.";
        } else if (model.downloadable && download_if_missing) {
            snapshot_.model_status = "Selected model is not installed: " + model_label(model.info);
            snapshot_.operation_status += " Downloading anonymously.";
            snapshot_.interpretation = snapshot_.decision.reason;
            should_download = true;
        } else if (model.downloadable) {
            snapshot_.operation_status += " Model is not installed.";
            return false;
        } else {
            snapshot_.operation_status = "That llama.cpp cache model is no longer available on disk.";
            return false;
        }
    }

    if (should_download) {
        return download_selected_model_unlocked();
    }
    return load_model_status_unlocked(model);
}

bool BackendSession::load_model_path(const std::string & path, const std::string & label) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    namespace fs = std::filesystem;

    std::error_code error;
    if (!fs::exists(path, error) || !fs::is_regular_file(path, error)) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_status = "Failed to load model: " + path;
        snapshot_.operation_status = "Model path must point to an existing regular GGUF file.";
        snapshot_.interpretation = "No model is loaded.";
        return false;
    }

    ModelStatus model;
    if (!describe_local_model_path(path, model.info)) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_status = "Failed to load model: " + path;
        snapshot_.operation_status = "Model path must be a recognizable Llama 3 8B GGUF with 4-bit quantization or higher.";
        snapshot_.interpretation = "No model is loaded.";
        return false;
    }
    if (!label.empty()) {
        model.info.quant = label;
    }
    model.path = path;
    model.cached = true;
    model.downloadable = false;
    model.catalog_model = false;
    return load_model_status_unlocked(model);
}

void BackendSession::apply_analysis_result_locked(const AnalysisResult & result, const std::string & source_label) {
    if (!result.ok) {
        snapshot_.operation_status = result.error;
        snapshot_.interpretation = "No score produced.";
        snapshot_.score_text = "-";
        snapshot_.ai_probability = "-";
        snapshot_.token_count = std::to_string(result.tokens);
        snapshot_.elapsed = "-";
        snapshot_.speed = "-";
    } else {
        snapshot_.operation_status = "Analysis complete.";
        snapshot_.score_text = format_fixed(result.discrepancy, 4);
        snapshot_.ai_probability = format_percent(ai_probability_from_score(result.discrepancy));
        snapshot_.interpretation = interpret_score(result.discrepancy);
        snapshot_.token_count = std::to_string(result.tokens);
        snapshot_.elapsed = format_fixed(result.elapsed_seconds, 2) + " s";
        snapshot_.speed = format_fixed(result.tokens_per_second, 2) + " tokens/sec";
    }
    snapshot_.input_source = source_label;
}

AnalysisResult BackendSession::analyze_text(const std::string & text) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    AnalysisResult             result;
    LlamaStatePtr              llama;
    ModelInfo                  loaded_model_info;
    std::string                loaded_model_path;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!snapshot_.model_ready || !llama_) {
            result.error = "Model is not ready yet.";
            apply_analysis_result_locked(result, "Pasted text");
            return result;
        }
        snapshot_.operation_status = "Running detection on pasted text...";
        snapshot_.input_source = "Pasted text";
        snapshot_.score_text = "-";
        snapshot_.ai_probability = "-";
        snapshot_.interpretation = "Running inference and scoring.";
        snapshot_.token_count = "-";
        snapshot_.elapsed = "-";
        snapshot_.speed = "measuring...";
        llama = llama_;
        loaded_model_info = loaded_model_info_;
        loaded_model_path = snapshot_.loaded_model_path;
    }

    result = analyze_text_detailed(*llama, text, config_.n_ctx);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        apply_analysis_result_locked(result, "Pasted text");
    }
    if (result.ok && !loaded_model_path.empty()) {
        std::string save_error;
        save_last_used_model(loaded_model_info, loaded_model_path, save_error);
    }
    return result;
}

AnalysisResult BackendSession::analyze_file(const std::string & path) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    namespace fs = std::filesystem;

    const std::string source_label = "File: " + fs::path(path).filename().string();
    AnalysisResult    result;
    LlamaStatePtr     llama;
    std::string       input_text;
    ModelInfo         loaded_model_info;
    std::string       loaded_model_path;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!snapshot_.model_ready || !llama_) {
            result.error = "Model is not ready yet.";
            apply_analysis_result_locked(result, source_label);
            return result;
        }
        snapshot_.operation_status = "Reading file and running detection...";
        snapshot_.input_source = source_label;
        snapshot_.score_text = "-";
        snapshot_.ai_probability = "-";
        snapshot_.interpretation = "Running inference and scoring.";
        snapshot_.token_count = "-";
        snapshot_.elapsed = "-";
        snapshot_.speed = "measuring...";
        llama = llama_;
        loaded_model_info = loaded_model_info_;
        loaded_model_path = snapshot_.loaded_model_path;
    }

    std::error_code path_error;
    if (!fs::exists(path, path_error) || !fs::is_regular_file(path, path_error)) {
        result.error = "Input must be an existing regular file.";
    } else if (!is_supported_input_file(path)) {
        result.error = "Only .md and .txt files are supported for now.";
    } else if (!read_file_to_string(path, input_text)) {
        result.error = "Failed to read input file.";
    } else {
        result = analyze_text_detailed(*llama, input_text, config_.n_ctx);
    }

    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        apply_analysis_result_locked(result, source_label);
    }
    if (result.ok && !loaded_model_path.empty()) {
        std::string save_error;
        save_last_used_model(loaded_model_info, loaded_model_path, save_error);
    }
    return result;
}

AnalysisResult BackendSession::analyze_input(const DetectionInput & input) {
    if (input.kind == DetectionInputKind::File) {
        return analyze_file(input.value);
    }
    return analyze_text(input.value);
}
