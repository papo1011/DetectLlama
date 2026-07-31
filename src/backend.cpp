#include "../include/backend.h"

#include "../include/io.h"
#include "../include/llama_state.h"
#include "../include/signals.h"

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

void apply_model_calibration(AnalysisResult & result) {
    const auto & model = detectllama_model();
    if (!result.ok || model.threshold_context <= 0 || model.calibration_id.empty() ||
        result.context_length != model.threshold_context) {
        return;
    }

    result.calibrated   = true;
    result.threshold    = model.threshold;
    result.predicted_ai = result.discrepancy >= result.threshold;
    result.calibration_id = model.calibration_id;
}

}  // namespace

std::string format_fixed(const double value, const int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string score_direction(const double score) {
    if (score > 0.0) {
        return "model-like ↑";
    }
    if (score < 0.0) {
        return "less model-like ↓";
    }
    return "neutral";
}

std::string interpret_score(const double score, const std::string & warning) {
    std::string interpretation;
    if (score > 0.0) {
        interpretation = "Positive discrepancy: more model-like.";
    } else if (score < 0.0) {
        interpretation = "Negative discrepancy: less model-like.";
    } else {
        interpretation = "Zero discrepancy: no directional signal.";
    }
    interpretation += " Calibrate a threshold before classifying.";
    if (!warning.empty()) {
        return warning + " " + interpretation;
    }
    return interpretation;
}

std::string interpret_result(const AnalysisResult & result) {
    if (!result.calibrated) {
        return interpret_score(result.discrepancy, result.warning);
    }

    std::string interpretation =
        result.predicted_ai ? "Above the calibrated threshold: AI-like." : "Below the calibrated threshold: human-like.";
    interpretation += " Experimental Ghostbuster essay calibration; not an AI probability.";
    if (!result.warning.empty()) {
        return result.warning + " " + interpretation;
    }
    return interpretation;
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
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
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
    static const std::vector<std::string> commands = { "/download", "/path" };
    const std::string                     trimmed  = lower_copy(trim_copy(value));
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
    if (command == "/download") {
        return "download the fixed Q4_0 model";
    }
    if (command == "/path") {
        return "load a local .txt or .md file";
    }
    return "";
}

std::string normalize_dropped_path(const std::string & raw_path) {
    std::string path           = trim_copy(raw_path);
    const auto  first_line_end = path.find_first_of("\r\n");
    if (first_line_end != std::string::npos) {
        path = trim_copy(path.substr(0, first_line_end));
    }

    if (path.size() >= 2 &&
        ((path.front() == '"' && path.back() == '"') || (path.front() == '\'' && path.back() == '\''))) {
        path = path.substr(1, path.size() - 2);
    }

    constexpr std::string_view file_scheme = "file://";
    if (path.rfind(file_scheme, 0) == 0) {
        path                                 = path.substr(file_scheme.size());
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
        result.action  = PromptAction::Empty;
        result.message = "Write /download, /path <file>, or paste text before analyzing.";
        return result;
    }

    if (starts_with_slash_command(trimmed_prompt, "/download")) {
        result.action = PromptAction::DownloadModel;
        return result;
    }

    if (trimmed_prompt.front() == '/' && !starts_with_slash_command(trimmed_prompt, "/path")) {
        result.action  = PromptAction::UnknownCommand;
        result.message = "Unknown command. Available commands: /download and /path <file>.";
        return result;
    }

    if (starts_with_slash_command(trimmed_prompt, "/path")) {
        const std::string path = normalize_dropped_path(slash_command_argument(trimmed_prompt, "/path"));
        if (path.empty()) {
            result.action  = PromptAction::UnknownCommand;
            result.message = "Use /path followed by a .md or .txt file path.";
            return result;
        }

        result.action             = PromptAction::LoadFile;
        result.input.kind         = DetectionInputKind::File;
        result.input.value        = path;
        result.input.source_label = "File: " + fs::path(path).filename().string();
        return result;
    }

    const std::string possible_path = normalize_dropped_path(trimmed_prompt);
    std::error_code   path_error;
    if (possible_path.find_first_of("\r\n") == std::string::npos && possible_path.size() < 4096 &&
        fs::exists(possible_path, path_error) && fs::is_regular_file(possible_path, path_error)) {
        result.action             = PromptAction::LoadFile;
        result.input.kind         = DetectionInputKind::File;
        result.input.value        = possible_path;
        result.input.source_label = "File: " + fs::path(possible_path).filename().string();
        return result;
    }

    result.action             = PromptAction::Analyze;
    result.input.kind         = DetectionInputKind::Text;
    result.input.value        = raw_prompt;
    result.input.source_label = "Pasted text";
    return result;
}

void BackendSession::LlamaStateDeleter::operator()(LlamaState * state) const {
    if (state) {
        free_llama_state(*state);
        delete state;
    }
}

BackendSession::BackendSession(AppConfig config) :
    config_(std::move(config)),
    llama_(LlamaStatePtr(new LlamaState{}, LlamaStateDeleter{})) {}

BackendSession::~BackendSession() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    LlamaStatePtr               old_llama;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        old_llama             = std::move(llama_);
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
    snapshot_.operation_status = snapshot_.model_ready ?
                                     "Type / for commands, /path <file>, or paste text directly into the prompt." :
                                     "Waiting for model.";
    snapshot_.score_text       = "-";
    snapshot_.score_direction  = "-";
    snapshot_.classification   = "-";
    snapshot_.threshold_text   = "-";
    snapshot_.input_source     = "-";
    snapshot_.interpretation = snapshot_.model_ready ? "Ready to analyze files or pasted text." : "Waiting for model.";
    snapshot_.token_count    = "-";
    snapshot_.elapsed        = "-";
    snapshot_.speed          = "-";
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
        snapshot_.model_status     = "Profiling this machine and checking the llama.cpp cache...";
        snapshot_.operation_status = "Type / for commands, /path <file>, or paste text to detect.";
        snapshot_.interpretation   = "Waiting for model.";
    }

    const ModelState state = inspect_model_state();
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.profile_summary = hardware_summary(state.hardware);
        if (!state.cached) {
            snapshot_.model_status     = model_label() + " is not installed.";
            snapshot_.operation_status = "Use /download to install the fixed Q4_0 model (about 4.6 GiB).";
            snapshot_.interpretation   = state.fits ? "The model fits this machine and is ready to download." :
                                                      "The model may not fit the available memory or disk space.";
        }
    }

    if (state.cached) {
        load_model_unlocked(state.path);
    }
}

bool BackendSession::load_model_unlocked(const std::string & path) {
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_ready = false;
        snapshot_.loaded_model_quant.clear();
        snapshot_.loaded_model_path.clear();
        snapshot_.model_status     = "Loading " + model_label() + " from local storage...";
        snapshot_.operation_status = "Model loading is running in the background.";
        snapshot_.interpretation   = "Waiting for model.";
    }

    ensure_backend_initialized();
    install_signal_handlers();

    auto       next = LlamaStatePtr(new LlamaState{}, LlamaStateDeleter{});
    const bool ok   = setup_llama(*next, path, config_.use_gpu);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (ok) {
            llama_                       = next;
            snapshot_.model_ready        = true;
            snapshot_.loaded_model_quant = model_label();
            snapshot_.loaded_model_path  = path;
            snapshot_.model_status       = "Model ready: " + model_label();
            snapshot_.operation_status   = "Type / for commands, /path <file>, or paste text directly into the prompt.";
            snapshot_.interpretation     = "Ready to analyze files or pasted text.";
        } else {
            snapshot_.model_status     = "Failed to load model: " + path;
            snapshot_.operation_status = "Check the Q4_0 GGUF and try again.";
            snapshot_.interpretation   = "No model is loaded.";
        }
    }
    return ok;
}

bool BackendSession::download_and_load_model() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    const std::string existing_path = cached_model_path();
    if (!existing_path.empty()) {
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            if (snapshot_.model_ready && snapshot_.loaded_model_path == existing_path) {
                snapshot_.operation_status = model_label() + " is already downloaded and loaded.";
                return true;
            }
        }
        return load_model_unlocked(existing_path);
    }

    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_ready      = false;
        snapshot_.model_status     = "Downloading " + model_label() + " anonymously from Hugging Face...";
        snapshot_.operation_status = "Download is running. The terminal stays inside DetectLlama.";
        snapshot_.interpretation   = "Waiting for download.";
    }

    std::string output_path;
    std::string error;
    if (!download_model(output_path, error)) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_status     = error;
        snapshot_.operation_status = "Download failed. Check network access and available disk space.";
        snapshot_.interpretation   = "No model is loaded.";
        return false;
    }
    return load_model_unlocked(output_path);
}

bool BackendSession::load_model_path(const std::string & path) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    if (!is_detectllama_model_path(path)) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        snapshot_.model_status = "Failed to load model: " + path;
        snapshot_.operation_status =
            "Model path must point to Meta-Llama-3-8B.Q4_0.gguf, the only supported GGUF.";
        snapshot_.interpretation = "No model is loaded.";
        return false;
    }
    return load_model_unlocked(path);
}

void BackendSession::apply_analysis_result_locked(const AnalysisResult & result, const std::string & source_label) {
    if (!result.ok) {
        snapshot_.operation_status = result.error;
        snapshot_.interpretation   = "No score produced.";
        snapshot_.score_text       = "-";
        snapshot_.score_direction  = "-";
        snapshot_.classification   = "-";
        snapshot_.threshold_text   = "-";
        snapshot_.token_count      = std::to_string(result.tokens);
        snapshot_.elapsed          = "-";
        snapshot_.speed            = "-";
    } else {
        snapshot_.operation_status = result.windows > 1 ?
                                         "Analysis complete across " + std::to_string(result.windows) + " context windows." :
                                         "Analysis complete.";
        snapshot_.score_text       = format_fixed(result.discrepancy, 4);
        snapshot_.score_direction  = score_direction(result.discrepancy);
        snapshot_.classification   = result.calibrated ? (result.predicted_ai ? "AI-like" : "human-like") : "uncalibrated";
        snapshot_.threshold_text   = result.calibrated ? format_fixed(result.threshold, 4) : "-";
        snapshot_.interpretation   = interpret_result(result);
        snapshot_.token_count      = std::to_string(result.tokens);
        snapshot_.elapsed          = format_fixed(result.elapsed_seconds, 2) + " s";
        snapshot_.speed            = format_fixed(result.tokens_per_second, 2) + " tokens/sec";
    }
    snapshot_.input_source = source_label;
}

// Captures the loaded model under the state lock so inference can run without
// holding UI state.
bool BackendSession::prepare_analysis_locked(AnalysisResult &      result,
                                             ActiveModelSnapshot & model,
                                             const std::string &   source_label,
                                             const std::string &   operation_status) {
    if (!snapshot_.model_ready || !llama_) {
        result.error = "Model is not ready yet.";
        apply_analysis_result_locked(result, source_label);
        return false;
    }

    snapshot_.operation_status = operation_status;
    snapshot_.input_source     = source_label;
    snapshot_.score_text       = "-";
    snapshot_.score_direction  = "-";
    snapshot_.classification   = "-";
    snapshot_.threshold_text   = "-";
    snapshot_.interpretation   = "Running inference and scoring.";
    snapshot_.token_count      = "-";
    snapshot_.elapsed          = "-";
    snapshot_.speed            = "measuring...";

    model.llama = llama_;
    return true;
}

AnalysisResult BackendSession::analyze_text(const std::string & text) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    AnalysisResult              result;
    ActiveModelSnapshot         model;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!prepare_analysis_locked(result, model, "Pasted text", "Running detection on pasted text...")) {
            return result;
        }
    }

    install_signal_handlers();
    result = analyze_text_detailed(*model.llama, text);
    apply_model_calibration(result);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        apply_analysis_result_locked(result, "Pasted text");
    }
    return result;
}

AnalysisResult BackendSession::analyze_file(const std::string & path) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    namespace fs = std::filesystem;

    const std::string   source_label = "File: " + fs::path(path).filename().string();
    AnalysisResult      result;
    ActiveModelSnapshot model;
    std::string         input_text;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!prepare_analysis_locked(result, model, source_label, "Reading file and running detection...")) {
            return result;
        }
    }

    std::error_code path_error;
    if (!fs::exists(path, path_error) || !fs::is_regular_file(path, path_error)) {
        result.error = "Input must be an existing regular file.";
    } else if (!is_supported_input_file(path)) {
        result.error = "Only .md and .txt files are supported for now.";
    } else if (!read_file_to_string(path, input_text)) {
        result.error = "Failed to read input file.";
    } else {
        install_signal_handlers();
        result = analyze_text_detailed(*model.llama, input_text);
        apply_model_calibration(result);
    }

    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        apply_analysis_result_locked(result, source_label);
    }
    return result;
}

AnalysisResult BackendSession::analyze_input(const DetectionInput & input) {
    if (input.kind == DetectionInputKind::File) {
        return analyze_file(input.value);
    }
    return analyze_text(input.value);
}
