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
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
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
    std::string   operation_status = "Drop, paste, or type a .md/.txt file path after the model is ready.";
    std::string   input_path;
    std::string   score_text = "-";
    std::string   interpretation = "Waiting for model.";
    std::string   token_count = "-";
    std::string   elapsed = "-";
    std::string   speed = "-";
    std::string   profile_summary = "Detecting hardware...";
    bool          model_busy = true;
    bool          analysis_busy = false;
    bool          model_ready = false;
    int           animation_frame = 0;

    auto screen = ScreenInteractive::Fullscreen();

    auto refresh_model_cache_state = [&] {
        for (auto & model : decision.models) {
            model.path = cached_model_path(config.model_repo, model.info.filename);
            model.cached = !model.path.empty();
        }
    };

    auto selected_model = [&]() -> ModelStatus * {
        if (!decision_ready || decision.models.empty()) {
            return nullptr;
        }
        selected_model_index = std::clamp(selected_model_index, 0, static_cast<int>(decision.models.size()) - 1);
        return &decision.models[selected_model_index];
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
                operation_status = "Drop, paste, or type a .md/.txt file path, then choose Analyze.";
                interpretation = "Ready to analyze local Markdown or text files.";
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
                    operation_status = "Choose Download to install it with llama-cli -hf, or inspect the other quantizations.";
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
            model_busy = true;
            model_ready = false;
            model_status = "Downloading " + model.info.quant + " with llama-cli -hf...";
            operation_status = "Download is running. The terminal stays inside DetectLlama.";
            interpretation = "Waiting for download.";
        }

        start_model_task([&, model] {
            std::string output_path;
            std::string error;
            const bool ok = download_model_with_llama_cli(config, model.info, output_path, error);
            ModelStatus downloaded = model;
            downloaded.path = output_path;
            downloaded.cached = ok;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                refresh_model_cache_state();
                if (!ok) {
                    model_busy = false;
                    model_status = error;
                    operation_status = "Download failed. Check llama-cli availability and network access.";
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
                operation_status = "Selected model is not installed. Choose Download first.";
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

    InputOption input_options;
    input_options.content = &input_path;
    input_options.placeholder = "drop or paste a .md/.txt file path";
    input_options.multiline = false;
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
            operation_status = "Analyzing file. Please wait...";
            score_text = "-";
            interpretation = "Running inference and scoring.";
            token_count = "-";
            elapsed = "-";
            speed = "measuring...";
            llama_for_analysis = llama;
        }

        const std::string path = normalize_dropped_path(input_path);
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            input_path = path;
        }
        if (analysis_worker.joinable()) {
            analysis_worker.join();
        }

        analysis_worker = std::thread([&, path, llama_for_analysis] {
            if (!fs::exists(path) || !fs::is_regular_file(path)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    operation_status = "Input must be an existing regular file.";
                    interpretation = "No analysis run.";
                    speed = "-";
                    analysis_busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            if (!is_supported_input_file(path)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    operation_status = "Only .md and .txt files are supported for now.";
                    interpretation = "No analysis run.";
                    speed = "-";
                    analysis_busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            std::string input_text;
            if (!read_file_to_string(path, input_text)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    operation_status = "Failed to read input file.";
                    interpretation = "No analysis run.";
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
                    token_count = std::to_string(result.tokens);
                    elapsed = "-";
                    speed = "-";
                } else {
                    operation_status = "Analysis complete.";
                    score_text = format_fixed(result.discrepancy, 4);
                    interpretation = interpret_score(result.discrepancy);
                    token_count = std::to_string(result.tokens);
                    elapsed = format_fixed(result.elapsed_seconds, 2) + " s";
                    speed = format_fixed(result.tokens_per_second, 2) + " tokens/sec";
                }
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
        input_path.clear();
        operation_status = model_ready ? "Drop, paste, or type a .md/.txt file path, then choose Analyze."
                                       : "Waiting for model.";
        score_text = "-";
        interpretation = model_ready ? "Ready to analyze local Markdown or text files." : "Waiting for model.";
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

    auto prev_button = Button("Previous", select_previous_model);
    auto next_button = Button("Next", select_next_model);
    auto download_button = Button("Download", download_selected_model);
    auto load_button = Button("Load", load_selected_cached_model);
    auto analyze_button = Button("Analyze", analyze);
    auto clear_button = Button("Clear", clear);
    auto quit_button = Button("Quit", quit);

    auto buttons = Container::Horizontal({
        prev_button,
        next_button,
        download_button,
        load_button,
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
        std::string   model_status_view;
        std::string   operation_status_view;
        std::string   score_view;
        std::string   interpretation_view;
        std::string   token_count_view;
        std::string   elapsed_view;
        std::string   speed_view;
        std::string   profile_summary_view;
        bool          busy_view;
        bool          model_ready_view;
        int           animation_frame_view;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            decision_view = decision;
            decision_ready_view = decision_ready;
            selected_model_index_view = selected_model_index;
            loaded_model_quant_view = loaded_model_quant;
            model_status_view = model_status;
            operation_status_view = operation_status;
            score_view = score_text;
            interpretation_view = interpretation;
            token_count_view = token_count;
            elapsed_view = elapsed;
            speed_view = speed;
            profile_summary_view = profile_summary;
            busy_view = model_busy || analysis_busy;
            model_ready_view = model_ready;
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
                if (model.info.quant == loaded_model_quant_view) {
                    label += "  loaded";
                }

                auto row = hbox({
                    text(index == selected_model_index_view ? "> " : "  "),
                    text(label) | bold,
                    filler(),
                    text(std::to_string(model.info.size_mb) + " MiB"),
                    text("  "),
                    text(model.cached ? "cached" : "missing"),
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

        auto metrics = vbox({
                           hbox(text("Discrepancy") | bold, filler(), text(score_view)),
                           separator(),
                           hbox(text("Tokens"), filler(), text(token_count_view)),
                           hbox(text("Elapsed"), filler(), text(elapsed_view)),
                           hbox(text("Speed"), filler(), text(speed_view)),
                       }) |
                       border;

        auto model_panel = vbox({
                               text("Models") | bold,
                               text(profile_summary_view) | dim,
                               separator(),
                               vbox(std::move(model_rows)),
                               separator(),
                               paragraph(decision_ready_view ? decision_view.reason : "DetectLlama is choosing the best Falcon 7B GGUF for this machine."),
                           }) |
                           border | flex;

        auto analysis_panel = vbox({
                                  text("File path") | bold,
                                  input->Render() | border,
                                  paragraph(operation_status_view),
                                  separator(),
                                  text("Result") | bold,
                                  paragraph(interpretation_view),
                                  filler(),
                                  buttons->Render(),
                              }) |
                              border | flex;

        auto body = hbox({
                        model_panel | size(WIDTH, GREATER_THAN, 44),
                        separator(),
                        analysis_panel,
                        separator(),
                        metrics | size(WIDTH, GREATER_THAN, 30),
                    }) |
                    flex;

        return vbox({
                   hbox(text("DetectLlama") | bold, filler(), text(model_ready_view ? "model ready" : "model setup") | dim),
                   separator(),
                   status_line,
                   separator(),
                   body,
                   separator(),
                   text("Supported input: .md and .txt. Dragging a file into most terminals pastes its path.") | dim,
               }) |
               border | flex;
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
