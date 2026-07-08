#include "../include/tui.h"

#include "../include/backend.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

int run_tui(const AppConfig & config) {
    using namespace ftxui;

    std::mutex       ui_mutex;
    BackendSession   backend(config);
    std::thread      model_worker;
    std::thread      analysis_worker;
    std::thread      ticker;
    std::atomic_bool keep_ticking = true;

    std::string prompt;
    bool        model_busy = true;
    bool        analysis_busy = false;
    bool        slash_menu_open = false;
    bool        model_picker_open = false;
    int         slash_menu_index = 0;
    int         animation_frame = 0;

    auto screen = ScreenInteractive::Fullscreen();

    auto busy = [&] {
        std::lock_guard<std::mutex> lock(ui_mutex);
        return model_busy || analysis_busy;
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

    auto start_model_task = [&](std::function<void()> task) {
        if (model_worker.joinable()) {
            model_worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            model_busy = true;
        }
        model_worker = std::thread([&, task = std::move(task)] {
            task();
            {
                std::lock_guard<std::mutex> lock(ui_mutex);
                model_busy = false;
            }
            screen.PostEvent(Event::Custom);
        });
    };

    auto open_models_picker = [&] {
        const auto snapshot = backend.snapshot();
        if (!snapshot.decision_ready || snapshot.decision.models.empty()) {
            backend.set_operation_status("Model list is not ready yet.");
            return;
        }
        if (busy()) {
            backend.set_operation_status("Please wait for the current operation to finish.");
            return;
        }

        backend.refresh_model_cache_state();
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            model_picker_open = true;
            slash_menu_open = false;
            prompt.clear();
        }
        backend.set_operation_status("Select a model. Enter loads cached models or downloads missing ones.");
        screen.PostEvent(Event::Custom);
    };

    auto run_model_query = [&](std::string query) {
        if (busy()) {
            backend.set_operation_status("Please wait for the current operation to finish.");
            return;
        }
        start_model_task([&, query = std::move(query)] { backend.load_model_by_query(query, true); });
    };

    auto activate_selected_model = [&] {
        if (busy()) {
            backend.set_operation_status("Please wait for the current operation to finish.");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            model_picker_open = false;
            slash_menu_open = false;
        }
        screen.PostEvent(Event::Custom);
        start_model_task([&] { backend.activate_selected_model(); });
    };

    auto apply_slash_command = [&](const std::string & command) {
        if (command == "/models") {
            open_models_picker();
            return;
        }

        if (command == "/path") {
            {
                std::lock_guard<std::mutex> lock(ui_mutex);
                prompt = "/path ";
                slash_menu_open = false;
            }
            backend.set_operation_status("Write or paste the file path after /path.");
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
            std::lock_guard<std::mutex> lock(ui_mutex);
            raw_prompt = prompt;
        }

        const PromptParseResult parsed = parse_prompt_input(raw_prompt);
        if (parsed.action == PromptAction::Empty || parsed.action == PromptAction::UnknownCommand) {
            backend.set_operation_status(parsed.message);
            return;
        }

        if (parsed.action == PromptAction::ModelPicker) {
            open_models_picker();
            return;
        }

        if (parsed.action == PromptAction::ModelQuery) {
            run_model_query(parsed.model_query);
            return;
        }

        if (busy()) {
            backend.set_operation_status("Please wait for the current operation to finish.");
            return;
        }
        if (!backend.snapshot().model_ready) {
            backend.set_operation_status("Model is not ready yet.");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            analysis_busy = true;
            if (parsed.input.kind == DetectionInputKind::File) {
                prompt = "/path " + parsed.input.value;
            }
        }

        if (analysis_worker.joinable()) {
            analysis_worker.join();
        }

        analysis_worker = std::thread([&, input = parsed.input] {
            backend.analyze_input(input);
            {
                std::lock_guard<std::mutex> lock(ui_mutex);
                analysis_busy = false;
            }
            screen.PostEvent(Event::Custom);
        });
    };

    auto clear = [&] {
        if (busy()) {
            backend.set_operation_status("Cannot clear while an operation is running.");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            prompt.clear();
            slash_menu_open = false;
            model_picker_open = false;
        }
        backend.clear_analysis();
        screen.PostEvent(Event::Custom);
    };

    auto quit = [&] {
        if (busy()) {
            backend.set_operation_status("An operation is running; wait for it to finish before quitting.");
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
        const BackendSnapshot snapshot = backend.snapshot();

        std::string prompt_view;
        bool        busy_view;
        bool        slash_menu_open_view;
        bool        model_picker_open_view;
        int         slash_menu_index_view;
        int         animation_frame_view;
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            prompt_view = prompt;
            busy_view = model_busy || analysis_busy;
            slash_menu_open_view = slash_menu_open;
            model_picker_open_view = model_picker_open;
            slash_menu_index_view = slash_menu_index;
            animation_frame_view = animation_frame;
        }

        Elements model_rows;
        if (!snapshot.decision_ready) {
            model_rows.push_back(text("Profiling hardware and cache...") | dim);
        } else {
            for (int index = 0; index < static_cast<int>(snapshot.decision.models.size()); ++index) {
                const auto & model = snapshot.decision.models[index];
                std::string label = model.info.quant;
                if (model.recommended) {
                    label += "  recommended";
                }
                if (!model.catalog_model) {
                    label += "  local";
                }
                if (!snapshot.loaded_model_path.empty() && model.path == snapshot.loaded_model_path) {
                    label += "  loaded";
                }

                const std::string cache_status = model.catalog_model ? (model.cached ? "cached" : "missing")
                                                                      : (model.cached ? "llama.cpp" : "missing");
                auto row = hbox({
                    text(index == snapshot.selected_model_index ? "> " : "  "),
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
                if (index == snapshot.selected_model_index) {
                    row |= inverted;
                }
                model_rows.push_back(row);
            }
        }

        auto spinner_or_space = busy_view ? spinner(6, static_cast<std::size_t>(animation_frame_view)) : text(" ");
        auto status_line = hbox({
            spinner_or_space,
            text(" "),
            text(snapshot.model_status) | (busy_view ? color(Color::Yellow) : color(Color::White)),
        });

        std::string selected_model_label = "-";
        if (snapshot.decision_ready && !snapshot.decision.models.empty()) {
            const int index = std::clamp(snapshot.selected_model_index, 0, static_cast<int>(snapshot.decision.models.size()) - 1);
            selected_model_label = snapshot.decision.models[index].info.quant;
        }
        const std::string loaded_model_label = snapshot.loaded_model_quant.empty() ? "-" : snapshot.loaded_model_quant;

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
                                hbox(text("Detection") | bold, filler(), text(snapshot.input_source) | dim),
                                paragraph(snapshot.interpretation),
                                separator(),
                                paragraph(snapshot.operation_status) | color(Color::Cyan),
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
                           hbox(text("AI probability"), filler(), text(snapshot.ai_probability) | bold),
                           hbox(text("Tokens/sec"), filler(), text(snapshot.speed)),
                           hbox(text("Tokens"), filler(), text(snapshot.token_count)),
                           hbox(text("Elapsed"), filler(), text(snapshot.elapsed)),
                           hbox(text("Score"), filler(), text(snapshot.score_text)),
                           separator(),
                           text("Model") | bold,
                           hbox(text("Loaded"), filler(), text(loaded_model_label)),
                           hbox(text("Selected"), filler(), text(selected_model_label)),
                           paragraph(snapshot.model_status) | (busy_view ? color(Color::Yellow) : color(Color::White)),
                           separator(),
                           text("Machine") | bold,
                           paragraph(snapshot.profile_summary) | dim,
                           separator(),
                           paragraph(snapshot.decision_ready ? snapshot.decision.reason
                                                             : "DetectLlama is choosing the best Falcon 7B GGUF for this machine.") |
                               dim,
                       }) |
                       border;

        auto body = hbox({
                        main_panel,
                        separator(),
                        sidebar | size(WIDTH, EQUAL, 38),
                    }) |
                    flex;

        return vbox({
                   hbox(text("DetectLlama") | bold, filler(), text(snapshot.model_ready ? "model ready" : "model setup") | dim),
                   separator(),
                   status_line,
                   separator(),
                   body,
               }) |
               border | flex;
    });

    renderer |= CatchEvent([&](Event event) {
        bool model_picker_active = false;
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            model_picker_active = model_picker_open;
        }

        if (model_picker_active) {
            if (event == Event::Escape) {
                {
                    std::lock_guard<std::mutex> lock(ui_mutex);
                    model_picker_open = false;
                }
                backend.set_operation_status("Model selector closed.");
                screen.PostEvent(Event::Custom);
                return true;
            }
            if (event == Event::ArrowDown || event == Event::Tab || event == Event::Character('j')) {
                backend.select_next_model();
                screen.PostEvent(Event::Custom);
                return true;
            }
            if (event == Event::ArrowUp || event == Event::TabReverse || event == Event::Character('k')) {
                backend.select_previous_model();
                screen.PostEvent(Event::Custom);
                return true;
            }
            if (event == Event::Return) {
                activate_selected_model();
                return true;
            }
            if (event != Event::Custom) {
                return true;
            }
        }

        std::string prompt_snapshot;
        int         slash_index_snapshot;
        {
            std::lock_guard<std::mutex> lock(ui_mutex);
            prompt_snapshot = prompt;
            slash_index_snapshot = slash_menu_index;
        }
        const auto matches = slash_command_matches(prompt_snapshot);
        if (matches.empty()) {
            return false;
        }

        if (event == Event::Escape) {
            {
                std::lock_guard<std::mutex> lock(ui_mutex);
                slash_menu_open = false;
            }
            backend.set_operation_status("Command menu closed.");
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::ArrowDown || event == Event::Tab || event == Event::Character('j')) {
            std::lock_guard<std::mutex> lock(ui_mutex);
            slash_menu_index = (slash_menu_index + 1) % static_cast<int>(matches.size());
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::ArrowUp || event == Event::TabReverse || event == Event::Character('k')) {
            std::lock_guard<std::mutex> lock(ui_mutex);
            slash_menu_index = (slash_menu_index + static_cast<int>(matches.size()) - 1) % static_cast<int>(matches.size());
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::Return) {
            const int index = std::clamp(slash_index_snapshot, 0, static_cast<int>(matches.size()) - 1);
            apply_slash_command(matches[index]);
            return true;
        }

        return false;
    });

    ticker = std::thread([&] {
        while (keep_ticking) {
            {
                std::lock_guard<std::mutex> lock(ui_mutex);
                ++animation_frame;
            }
            screen.PostEvent(Event::Custom);
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
    });

    start_model_task([&] { backend.initialize(); });
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
    return 0;
}
