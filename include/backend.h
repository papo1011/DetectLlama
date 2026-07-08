#pragma once

#include "./app_config.h"
#include "./detect.h"
#include "./model_manager.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

enum class DetectionInputKind {
    Text,
    File,
};

struct DetectionInput {
    DetectionInputKind kind = DetectionInputKind::Text;
    std::string        value;
    std::string        source_label = "Pasted text";
};

enum class PromptAction {
    Empty,
    ModelPicker,
    ModelQuery,
    Analyze,
    UnknownCommand,
};

struct PromptParseResult {
    PromptAction  action = PromptAction::Empty;
    DetectionInput input;
    std::string    model_query;
    std::string    message;
};

struct BackendSnapshot {
    ModelDecision decision;
    bool          decision_ready = false;
    int           selected_model_index = 0;
    std::string   loaded_model_quant;
    std::string   loaded_model_path;
    std::string   model_status = "Profiling this machine and checking the llama.cpp cache...";
    std::string   operation_status = "Type / for commands, /path <file>, or paste text to detect.";
    std::string   input_source = "-";
    std::string   score_text = "-";
    std::string   ai_probability = "-";
    std::string   interpretation = "Waiting for model.";
    std::string   token_count = "-";
    std::string   elapsed = "-";
    std::string   speed = "-";
    std::string   profile_summary = "Detecting hardware...";
    bool          model_ready = false;
};

std::string format_fixed(double value, int precision);
double      ai_probability_from_score(double score);
std::string format_percent(double probability);
std::string interpret_score(double score);

std::string trim_copy(const std::string & value);
std::string lower_copy(std::string value);
bool        starts_with_slash_command(const std::string & value, std::string_view command);
std::string slash_command_argument(const std::string & value, std::string_view command);
std::vector<std::string> slash_command_matches(const std::string & value);
std::string command_description(const std::string & command);
std::string normalize_dropped_path(const std::string & raw_path);
bool        is_supported_input_file(const std::filesystem::path & path);

PromptParseResult parse_prompt_input(const std::string & raw_prompt);

class BackendSession {
public:
    explicit BackendSession(AppConfig config);
    ~BackendSession();

    BackendSession(const BackendSession &) = delete;
    BackendSession & operator=(const BackendSession &) = delete;

    BackendSnapshot snapshot() const;

    void initialize();
    void set_operation_status(const std::string & status);
    void clear_analysis();

    bool refresh_model_cache_state();
    bool select_model_index(int index);
    void select_previous_model();
    void select_next_model();

    bool load_selected_model();
    bool activate_selected_model();
    bool load_model_by_query(const std::string & query, bool download_if_missing);
    bool load_model_path(const std::string & path, const std::string & label);

    AnalysisResult analyze_text(const std::string & text);
    AnalysisResult analyze_file(const std::string & path);
    AnalysisResult analyze_input(const DetectionInput & input);

private:
    struct LlamaStateDeleter {
        void operator()(LlamaState * state) const;
    };

    using LlamaStatePtr = std::shared_ptr<LlamaState>;

    void        ensure_backend_initialized();
    bool        load_model_status_unlocked(const ModelStatus & model);
    bool        download_selected_model_unlocked();
    ModelStatus selected_model_locked() const;
    void        refresh_model_cache_state_locked();
    void        apply_analysis_result_locked(const AnalysisResult & result, const std::string & source_label);
    void        reset_analysis_fields_locked();

    AppConfig config_;

    mutable std::mutex state_mutex_;
    std::mutex         operation_mutex_;

    BackendSnapshot snapshot_;
    LlamaStatePtr    llama_;
    bool             backend_initialized_ = false;
};
