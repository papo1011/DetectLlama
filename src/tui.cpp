#include "../include/tui.h"

#include "../include/detect.h"
#include "../include/io.h"
#include "../include/llama_state.h"
#include "../include/model_manager.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

namespace {

std::string format_fixed(const double value, const int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
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

double ai_probability_from_score(const double score) {
    const double probability = 1.0 / (1.0 + std::exp(1.35 * score));
    return std::clamp(probability, 0.0, 1.0);
}

std::string format_percent(const double probability) {
    return format_fixed(probability * 100.0, 1) + "%";
}

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

void free_llama_state(LlamaState & llama) {
    if (llama.ctx) {
        llama_free(llama.ctx);
        llama.ctx = nullptr;
    }
    if (llama.model) {
        llama_model_free(llama.model);
        llama.model = nullptr;
    }
    llama.vocab = nullptr;
}

}  // namespace

int run_tui(const AppConfig & config) {
    namespace fs = std::filesystem;
    using namespace ftxui;

    std::mutex              state_mutex;
    std::thread             model_worker;
    std::thread             analysis_worker;
    std::thread             ticker;
    std::atomic_bool        keep_ticking = true;
    std::shared_ptr<LlamaState> llama = std::make_shared<LlamaState>();
    bool                    backend_initialized = false;

    ModelDecision decision;
    bool          decision_ready = false;
    int           selected_model_index = 0;
    std::string   loaded_model_quant;
    std::string   loaded_model_path;
    std::string   model_status = "Profiling this machine and checking the llama.cpp cache...";
    std::string   operation_status = "Type / for commands, /path <file>, or paste text to detect.";
    std::string   prompt;
    std::string   input_source = "-";
    std::string   score_text = "-";
    std::string   ai_probability = "-";
    std::string   interpretation = "Waiting for model.";
    std::string   token_count = "-";
    std::string   elapsed = "-";
    std::string   speed = "-";
    std::string   profile_summary = "Detecting hardware...";
    bool          model_busy = true;
    bool          analysis_busy = false;
    bool          model_ready = false;
    bool          slash_menu_open = false;
    bool          model_picker_open = false;
    int           slash_menu_index = 0;
    int           animation_frame = 0;

    auto screen = ScreenInteractive::Fullscreen();

    auto refresh_model_cache_state = [&] {
        for (auto & model : decision.models) {
            if (model.catalog_model) {
                model.path = cached_model_path(config.model_repo, model.info.filename);
                model.cached = !model.path.empty();
            } else {
                std::error_code error;
                model.cached = !model.path.empty() && fs::exists(model.path, error);
            }
        }
    };

    auto selected_model = [&]() -> ModelStatus * {
        if (!decision_ready || decision.models.empty()) {
            return nullptr;
        }
        selected_model_index = std::clamp(selected_model_index, 0, static_cast<int>(decision.models.size()) - 1);
        return &decision.models[selected_model_index];
    };

    auto refresh_slash_menu_state = [&] {
        const auto matches = slash_command_matches(prompt);
        slash_menu_open = !model_picker_open && !matches.empty();
        if (slash_menu_open) {
            slash_menu_index = std::clamp(slash_menu_index, 0, static_cast<int>(matches.size()) - 1);
        } else {
            slash_menu_index = 0;
        }
    };

    auto load_model = [&](const ModelStatus model) {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            model_busy = true;
            model_ready = false;
            loaded_model_quant.clear();
            loaded_model_path.clear();
            model_status = "Loading " + model.info.quant + " from local cache...";
            operation_status = "Model loading is running in the background.";
            interpretation = "Waiting for model.";
        }

        if (!backend_initialized) {
            llama_backend_init();
            backend_initialized = true;
        }

        LlamaState next = {};
        const bool ok = setup_llama(next, model.path, config.use_gpu, config.n_ctx, config.n_batch);
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (ok) {
                free_llama_state(*llama);
                *llama = next;
                model_ready = true;
                loaded_model_quant = model.info.quant;
                loaded_model_path = model.path;
                model_status = "Model ready: " + model.info.quant;
                operation_status = "Type / for commands, /path <file>, or paste text directly into the prompt.";
                interpretation = "Ready to analyze files or pasted text.";
            } else {
                free_llama_state(next);
                model_status = "Failed to load model: " + model.path;
                operation_status = "Choose another cached model or download the recommended one.";
                interpretation = "No model is loaded.";
            }
            model_busy = false;
        }
        screen.PostEvent(Event::Custom);
    };

    auto start_model_task = [&](auto task) {
        if (model_worker.joinable()) {
            model_worker.join();
        }
        model_worker = std::thread(task);
    };

    auto profile_and_maybe_load = [&] {
        start_model_task([&] {
            ModelDecision next_decision = build_model_decision(config);
            ModelStatus   model_to_load;
            bool          should_load = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                decision = std::move(next_decision);
                decision_ready = true;
                selected_model_index = decision.recommended_index;
                const auto & hardware = decision.hardware;
                profile_summary = hardware.os_name + "/" + hardware.arch_name + " | " + std::to_string(hardware.cpu_cores) +
                                  " CPU cores | RAM " + std::to_string(hardware.available_ram_mb) + "/" +
                                  std::to_string(hardware.total_ram_mb) + " MiB | " + hardware.accelerator;
                if (!hardware.gpu_name.empty()) {
                    profile_summary += " " + hardware.gpu_name;
                }

                const auto & recommended = decision.models[decision.recommended_index];
                if (recommended.cached) {
                    model_to_load = recommended;
                    should_load = true;
                } else {
                    model_busy = false;
                    model_status = "Recommended model is not installed: " + recommended.info.quant;
                    operation_status = "Use /models to select and download a quantization.";
                    interpretation = decision.reason;
                }
            }
            screen.PostEvent(Event::Custom);
            if (should_load) {
                load_model(model_to_load);
            }
        });
    };

    auto download_selected_model = [&] {
        ModelStatus model;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!decision_ready || model_busy || analysis_busy) {
                operation_status = "Please wait for the current operation to finish.";
                return;
            }
            model = *selected_model();
            if (!model.downloadable) {
                operation_status = "This llama.cpp cache model is local-only and cannot be downloaded by DetectLlama.";
                return;
            }
            model_busy = true;
            model_ready = false;
            model_status = "Downloading " + model.info.quant + " anonymously from Hugging Face...";
            operation_status = "Download is running. The terminal stays inside DetectLlama.";
            interpretation = "Waiting for download.";
        }

        start_model_task([&, model] {
            std::string output_path;
            std::string error;
            const bool ok = download_public_model(config, model.info, output_path, error);
            ModelStatus downloaded = model;
            downloaded.path = output_path;
            downloaded.cached = ok;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                refresh_model_cache_state();
                if (!ok) {
                    model_busy = false;
                    model_status = error;
                    operation_status = "Download failed. Check network access and that the model repo is public.";
                    interpretation = "No model is loaded.";
                }
            }
            screen.PostEvent(Event::Custom);
            if (ok) {
                load_model(downloaded);
            }
        });
    };

    auto load_selected_cached_model = [&] {
        ModelStatus model;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!decision_ready || model_busy || analysis_busy) {
                operation_status = "Please wait for the current operation to finish.";
                return;
            }
            refresh_model_cache_state();
            model = *selected_model();
            if (!model.cached) {
                operation_status = "Selected model is not installed. Open /models and press Enter to download it.";
                return;
            }
        }
        start_model_task([&, model] { load_model(model); });
    };

    auto select_previous_model = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!decision_ready || decision.models.empty()) {
            return;
        }
        selected_model_index = (selected_model_index + static_cast<int>(decision.models.size()) - 1) %
                               static_cast<int>(decision.models.size());
        operation_status = "Selected " + decision.models[selected_model_index].info.quant + ".";
    };

    auto select_next_model = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!decision_ready || decision.models.empty()) {
            return;
        }
        selected_model_index = (selected_model_index + 1) % static_cast<int>(decision.models.size());
        operation_status = "Selected " + decision.models[selected_model_index].info.quant + ".";
    };

    auto open_models_picker = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!decision_ready || decision.models.empty()) {
            operation_status = "Model list is not ready yet.";
            return;
        }
        if (model_busy || analysis_busy) {
            operation_status = "Please wait for the current operation to finish.";
            return;
        }
        refresh_model_cache_state();
        model_picker_open = true;
        slash_menu_open = false;
        prompt.clear();
        operation_status = "Select a model. Enter loads cached models or downloads missing ones.";
    };

    auto handle_models_command = [&](const std::string & raw_prompt) {
        if (!starts_with_slash_command(raw_prompt, "/models")) {
            return false;
        }

        const std::string arg = lower_copy(slash_command_argument(raw_prompt, "/models"));
        if (arg.empty()) {
            open_models_picker();
            screen.PostEvent(Event::Custom);
            return true;
        }

        ModelStatus model_to_load;
        bool        should_load = false;
        bool        should_download = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!decision_ready || decision.models.empty()) {
                operation_status = "Model list is not ready yet.";
                return true;
            }
            if (model_busy || analysis_busy) {
                operation_status = "Please wait for the current operation to finish.";
                return true;
            }

            refresh_model_cache_state();
            int next_index = -1;
            for (int index = 0; index < static_cast<int>(decision.models.size()); ++index) {
                const auto & model = decision.models[index];
                if (lower_copy(model.info.quant).find(arg) != std::string::npos ||
                    lower_copy(model.info.filename).find(arg) != std::string::npos ||
                    std::to_string(model.info.rank) == arg) {
                    next_index = index;
                    break;
                }
            }
            if (next_index < 0) {
                operation_status = "Unknown model. Use /models to open the selector or /models <quant>.";
                return true;
            }

            selected_model_index = next_index;
            model_to_load = decision.models[selected_model_index];
            operation_status = "Selected " + model_to_load.info.quant + ".";
            if (model_to_load.cached) {
                operation_status += " Loading from cache.";
                should_load = true;
            } else if (model_to_load.downloadable) {
                model_status = "Selected model is not installed: " + model_to_load.info.quant;
                operation_status += " Downloading anonymously.";
                interpretation = decision.reason;
                should_download = true;
            } else {
                operation_status = "That llama.cpp cache model is no longer available on disk.";
            }
        }

        screen.PostEvent(Event::Custom);
        if (should_load) {
            start_model_task([&, model_to_load] { load_model(model_to_load); });
        } else if (should_download) {
            download_selected_model();
        }
        return true;
    };

    auto activate_selected_model = [&] {
        ModelStatus model;
        bool        cached = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!decision_ready || decision.models.empty()) {
                operation_status = "Model list is not ready yet.";
                return;
            }
            if (model_busy || analysis_busy) {
                operation_status = "Please wait for the current operation to finish.";
                return;
            }
            refresh_model_cache_state();
            model = *selected_model();
            cached = model.cached;
            if (!cached && !model.downloadable) {
                model_picker_open = false;
                slash_menu_open = false;
                operation_status = "That llama.cpp cache model is no longer available on disk.";
                return;
            }
            model_picker_open = false;
            slash_menu_open = false;
            operation_status = cached ? "Loading " + model.info.quant + " from cache."
                                      : "Downloading " + model.info.quant + " anonymously.";
        }

        screen.PostEvent(Event::Custom);
        if (cached) {
            start_model_task([&, model] { load_model(model); });
        } else {
            download_selected_model();
        }
    };

    auto apply_slash_command = [&](const std::string & command) {
        if (command == "/models") {
            open_models_picker();
            screen.PostEvent(Event::Custom);
            return;
        }

        if (command == "/path") {
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                prompt = "/path ";
                slash_menu_open = false;
                operation_status = "Write or paste the file path after /path.";
            }
            screen.PostEvent(Event::Custom);
        }
    };

    InputOption input_options;
    input_options.content = &prompt;
    input_options.placeholder = "Type / for commands, /path ./sample.txt, or paste text to detect";
    input_options.multiline = true;
    input_options.on_change = [&] { refresh_slash_menu_state(); };
    input_options.transform = [](InputState state) {
        state.element |= color(Color::White);
        if (state.is_placeholder) {
            state.element |= dim;
        }
        if (state.focused) {
            state.element |= bgcolor(Color::Black) | bold;
        } else if (state.hovered) {
            state.element |= bgcolor(Color::Black);
        }
        return state.element;
    };
    auto input = Input(input_options);

    auto analyze = [&] {
        std::string raw_prompt;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            raw_prompt = prompt;
        }

        const std::string trimmed_prompt = trim_copy(raw_prompt);
        if (trimmed_prompt.empty()) {
            std::lock_guard<std::mutex> lock(state_mutex);
            operation_status = "Write /models, /path <file>, or paste text before analyzing.";
            return;
        }

        if (handle_models_command(trimmed_prompt)) {
            return;
        }

        if (trimmed_prompt.front() == '/' && !starts_with_slash_command(trimmed_prompt, "/path")) {
            std::lock_guard<std::mutex> lock(state_mutex);
            operation_status = "Unknown command. Available commands: /models and /path <file>.";
            return;
        }

        bool        use_file = false;
        std::string path;
        std::string inline_text;
        std::string source_label;
        if (starts_with_slash_command(trimmed_prompt, "/path")) {
            path = normalize_dropped_path(slash_command_argument(trimmed_prompt, "/path"));
            if (path.empty()) {
                std::lock_guard<std::mutex> lock(state_mutex);
                operation_status = "Use /path followed by a .md or .txt file path.";
                return;
            }
            use_file = true;
            source_label = "File: " + fs::path(path).filename().string();
        } else {
            const std::string possible_path = normalize_dropped_path(trimmed_prompt);
            std::error_code   path_error;
            if (possible_path.find_first_of("\r\n") == std::string::npos && possible_path.size() < 4096 &&
                fs::exists(possible_path, path_error) && fs::is_regular_file(possible_path, path_error)) {
                use_file = true;
                path = possible_path;
                source_label = "File: " + fs::path(path).filename().string();
            } else {
                inline_text = trimmed_prompt;
                source_label = "Pasted text";
            }
        }

        std::shared_ptr<LlamaState> llama_for_analysis;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!model_ready) {
                operation_status = "Model is not ready yet.";
                return;
            }
            if (model_busy || analysis_busy) {
                operation_status = "Please wait for the current operation to finish.";
                return;
            }
            analysis_busy = true;
            operation_status = use_file ? "Reading file and running detection..." : "Running detection on pasted text...";
            input_source = source_label;
            score_text = "-";
            ai_probability = "-";
            interpretation = "Running inference and scoring.";
            token_count = "-";
            elapsed = "-";
            speed = "measuring...";
            llama_for_analysis = llama;
            if (use_file) {
                prompt = "/path " + path;
            }
        }

        if (analysis_worker.joinable()) {
            analysis_worker.join();
        }

        analysis_worker = std::thread([&, use_file, path, inline_text, source_label, llama_for_analysis] {
            std::string input_text = inline_text;
            std::error_code path_error;
            if (use_file && (!fs::exists(path, path_error) || !fs::is_regular_file(path, path_error))) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    operation_status = "Input must be an existing regular file.";
                    interpretation = "No analysis run.";
                    ai_probability = "-";
                    speed = "-";
                    analysis_busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            if (use_file && !is_supported_input_file(path)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    operation_status = "Only .md and .txt files are supported for now.";
                    interpretation = "No analysis run.";
                    ai_probability = "-";
                    speed = "-";
                    analysis_busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            if (use_file && !read_file_to_string(path, input_text)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    operation_status = "Failed to read input file.";
                    interpretation = "No analysis run.";
                    ai_probability = "-";
                    speed = "-";
                    analysis_busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            const AnalysisResult result = analyze_text_detailed(*llama_for_analysis, input_text, config.n_ctx);
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                if (!result.ok) {
                    operation_status = result.error;
                    interpretation = "No score produced.";
                    score_text = "-";
                    ai_probability = "-";
                    token_count = std::to_string(result.tokens);
                    elapsed = "-";
                    speed = "-";
                } else {
                    operation_status = "Analysis complete.";
                    score_text = format_fixed(result.discrepancy, 4);
                    ai_probability = format_percent(ai_probability_from_score(result.discrepancy));
                    interpretation = interpret_score(result.discrepancy);
                    token_count = std::to_string(result.tokens);
                    elapsed = format_fixed(result.elapsed_seconds, 2) + " s";
                    speed = format_fixed(result.tokens_per_second, 2) + " tokens/sec";
                }
                input_source = source_label;
                analysis_busy = false;
            }
            screen.PostEvent(Event::Custom);
        });
    };

    auto clear = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (model_busy || analysis_busy) {
            operation_status = "Cannot clear while an operation is running.";
            return;
        }
        prompt.clear();
        slash_menu_open = false;
        model_picker_open = false;
        operation_status = model_ready ? "Type / for commands, /path <file>, or paste text directly into the prompt."
                                       : "Waiting for model.";
        score_text = "-";
        ai_probability = "-";
        input_source = "-";
        interpretation = model_ready ? "Ready to analyze files or pasted text." : "Waiting for model.";
        token_count = "-";
        elapsed = "-";
        speed = "-";
    };

    auto quit = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (model_busy || analysis_busy) {
            operation_status = "An operation is running; wait for it to finish before quitting.";
            return;
        }
        screen.ExitLoopClosure()();
    };

    auto analyze_button = Button("Analyze", analyze);
    auto clear_button = Button("Clear", clear);
    auto quit_button = Button("Quit", quit);

    auto buttons = Container::Horizontal({
        analyze_button,
        clear_button,
        quit_button,
    });

    auto container = Container::Vertical({
        input,
        buttons,
    });

    auto renderer = Renderer(container, [&] {
        ModelDecision decision_view;
        bool          decision_ready_view;
        int           selected_model_index_view;
        std::string   loaded_model_quant_view;
        std::string   loaded_model_path_view;
        std::string   model_status_view;
        std::string   operation_status_view;
        std::string   score_view;
        std::string   ai_probability_view;
        std::string   input_source_view;
        std::string   interpretation_view;
        std::string   token_count_view;
        std::string   elapsed_view;
        std::string   speed_view;
        std::string   profile_summary_view;
        std::string   prompt_view;
        bool          busy_view;
        bool          model_ready_view;
        bool          slash_menu_open_view;
        bool          model_picker_open_view;
        int           slash_menu_index_view;
        int           animation_frame_view;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            decision_view = decision;
            decision_ready_view = decision_ready;
            selected_model_index_view = selected_model_index;
            loaded_model_quant_view = loaded_model_quant;
            loaded_model_path_view = loaded_model_path;
            model_status_view = model_status;
            operation_status_view = operation_status;
            score_view = score_text;
            ai_probability_view = ai_probability;
            input_source_view = input_source;
            interpretation_view = interpretation;
            token_count_view = token_count;
            elapsed_view = elapsed;
            speed_view = speed;
            profile_summary_view = profile_summary;
            prompt_view = prompt;
            busy_view = model_busy || analysis_busy;
            model_ready_view = model_ready;
            slash_menu_open_view = slash_menu_open;
            model_picker_open_view = model_picker_open;
            slash_menu_index_view = slash_menu_index;
            animation_frame_view = animation_frame;
        }

        Elements model_rows;
        if (!decision_ready_view) {
            model_rows.push_back(text("Profiling hardware and cache...") | dim);
        } else {
            for (int index = 0; index < static_cast<int>(decision_view.models.size()); ++index) {
                const auto & model = decision_view.models[index];
                std::string label = model.info.quant;
                if (model.recommended) {
                    label += "  recommended";
                }
                if (!model.catalog_model) {
                    label += "  local";
                }
                if (!loaded_model_path_view.empty() && model.path == loaded_model_path_view) {
                    label += "  loaded";
                }

                const std::string cache_status = model.catalog_model ? (model.cached ? "cached" : "missing")
                                                                      : (model.cached ? "llama.cpp" : "missing");
                auto row = hbox({
                    text(index == selected_model_index_view ? "> " : "  "),
                    text(label) | bold,
                    filler(),
                    text(std::to_string(model.info.size_mb) + " MiB"),
                    text("  "),
                    text(cache_status),
                });
                if (model.recommended) {
                    row |= color(Color::Green);
                }
                if (!model.fits) {
                    row |= dim;
                }
                if (index == selected_model_index_view) {
                    row |= inverted;
                }
                model_rows.push_back(row);
            }
        }

        auto spinner_or_space = busy_view ? spinner(6, static_cast<std::size_t>(animation_frame_view)) : text(" ");
        auto status_line = hbox({
            spinner_or_space,
            text(" "),
            text(model_status_view) | (busy_view ? color(Color::Yellow) : color(Color::White)),
        });

        std::string selected_model_label = "-";
        if (decision_ready_view && !decision_view.models.empty()) {
            selected_model_label = decision_view.models[std::clamp(selected_model_index_view, 0, static_cast<int>(decision_view.models.size()) - 1)].info.quant;
        }
        const std::string loaded_model_label = loaded_model_quant_view.empty() ? "-" : loaded_model_quant_view;

        const auto slash_matches = slash_command_matches(prompt_view);
        Elements   slash_rows;
        for (int index = 0; index < static_cast<int>(slash_matches.size()); ++index) {
            const auto & command = slash_matches[index];
            auto row = hbox({
                text(index == slash_menu_index_view ? "> " : "  "),
                text(command) | bold,
                filler(),
                text(command_description(command)) | dim,
            });
            if (index == slash_menu_index_view) {
                row |= inverted;
            }
            slash_rows.push_back(row);
        }

        Elements prompt_elements = {
            hbox(text("Prompt") | bold, filler(), text("/models  /path <file>  paste text") | dim),
            input->Render() | border | flex,
        };
        if (slash_menu_open_view && !slash_rows.empty()) {
            prompt_elements.push_back(vbox({
                                          text("Commands") | dim,
                                          vbox(std::move(slash_rows)),
                                      }) |
                                      border | size(WIDTH, LESS_THAN, 52));
        }
        auto prompt_panel = vbox(std::move(prompt_elements)) | flex;

        auto result_panel = vbox({
                                hbox(text("Detection") | bold, filler(), text(input_source_view) | dim),
                                paragraph(interpretation_view),
                                separator(),
                                paragraph(operation_status_view) | color(Color::Cyan),
                            }) |
                            border;

        auto model_picker_panel = vbox({
                                      hbox(text("Models") | bold, filler(), text("Enter load/download  Esc close") | dim),
                                      vbox(std::move(model_rows)) | yframe | size(HEIGHT, LESS_THAN, 12),
                                  }) |
                                  border;

        auto controls_panel = vbox({
                                  buttons->Render(),
                              }) |
                              border;

        Elements main_elements = {
            prompt_panel,
            separator(),
        };
        if (model_picker_open_view) {
            main_elements.push_back(model_picker_panel);
        } else {
            main_elements.push_back(result_panel);
        }
        main_elements.push_back(separator());
        main_elements.push_back(controls_panel);

        auto main_panel = vbox(std::move(main_elements)) | flex;

        auto sidebar = vbox({
                           text("Session") | bold,
                           separator(),
                           hbox(text("AI probability"), filler(), text(ai_probability_view) | bold),
                           hbox(text("Tokens/sec"), filler(), text(speed_view)),
                           hbox(text("Tokens"), filler(), text(token_count_view)),
                           hbox(text("Elapsed"), filler(), text(elapsed_view)),
                           hbox(text("Score"), filler(), text(score_view)),
                           separator(),
                           text("Model") | bold,
                           hbox(text("Loaded"), filler(), text(loaded_model_label)),
                           hbox(text("Selected"), filler(), text(selected_model_label)),
                           paragraph(model_status_view) | (busy_view ? color(Color::Yellow) : color(Color::White)),
                           separator(),
                           text("Machine") | bold,
                           paragraph(profile_summary_view) | dim,
                           separator(),
                           paragraph(decision_ready_view ? decision_view.reason : "DetectLlama is choosing the best Falcon 7B GGUF for this machine.") | dim,
                       }) |
                       border;

        auto body = hbox({
                        main_panel,
                        separator(),
                        sidebar | size(WIDTH, EQUAL, 38),
                    }) |
                    flex;

        return vbox({
                   hbox(text("DetectLlama") | bold, filler(), text(model_ready_view ? "model ready" : "model setup") | dim),
                   separator(),
                   status_line,
                   separator(),
                   body,
               }) |
               border | flex;
    });

    renderer |= CatchEvent([&](Event event) {
        bool should_activate_model = false;
        bool model_picker_active = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (model_picker_open) {
                model_picker_active = true;
                if (event == Event::Escape) {
                    model_picker_open = false;
                    operation_status = "Model selector closed.";
                    screen.PostEvent(Event::Custom);
                    return true;
                }
                if (!decision.models.empty() &&
                    (event == Event::ArrowDown || event == Event::Tab || event == Event::Character('j'))) {
                    selected_model_index = (selected_model_index + 1) % static_cast<int>(decision.models.size());
                    operation_status = "Selected " + decision.models[selected_model_index].info.quant + ".";
                    screen.PostEvent(Event::Custom);
                    return true;
                }
                if (!decision.models.empty() &&
                    (event == Event::ArrowUp || event == Event::TabReverse || event == Event::Character('k'))) {
                    selected_model_index = (selected_model_index + static_cast<int>(decision.models.size()) - 1) %
                                           static_cast<int>(decision.models.size());
                    operation_status = "Selected " + decision.models[selected_model_index].info.quant + ".";
                    screen.PostEvent(Event::Custom);
                    return true;
                }
                if (event == Event::Return) {
                    should_activate_model = true;
                }
            }
        }

        if (should_activate_model) {
            activate_selected_model();
            return true;
        }
        if (model_picker_active && event != Event::Custom) {
            return true;
        }

        std::string prompt_snapshot;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            prompt_snapshot = prompt;
        }
        const auto matches = slash_command_matches(prompt_snapshot);
        if (matches.empty()) {
            return false;
        }

        if (event == Event::Escape) {
            std::lock_guard<std::mutex> lock(state_mutex);
            slash_menu_open = false;
            operation_status = "Command menu closed.";
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::ArrowDown || event == Event::Tab || event == Event::Character('j')) {
            std::lock_guard<std::mutex> lock(state_mutex);
            slash_menu_index = (slash_menu_index + 1) % static_cast<int>(matches.size());
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::ArrowUp || event == Event::TabReverse || event == Event::Character('k')) {
            std::lock_guard<std::mutex> lock(state_mutex);
            slash_menu_index = (slash_menu_index + static_cast<int>(matches.size()) - 1) % static_cast<int>(matches.size());
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::Return) {
            const int index = std::clamp(slash_menu_index, 0, static_cast<int>(matches.size()) - 1);
            apply_slash_command(matches[index]);
            return true;
        }

        return false;
    });

    ticker = std::thread([&] {
        while (keep_ticking) {
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                ++animation_frame;
            }
            screen.PostEvent(Event::Custom);
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
    });

    profile_and_maybe_load();
    screen.Loop(renderer);

    keep_ticking = false;
    if (ticker.joinable()) {
        ticker.join();
    }
    if (model_worker.joinable()) {
        model_worker.join();
    }
    if (analysis_worker.joinable()) {
        analysis_worker.join();
    }
    free_llama_state(*llama);
    if (backend_initialized) {
        llama_backend_free();
    }
    return 0;
}
