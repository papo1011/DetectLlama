#include "../include/tui.h"

#include "../include/backend.h"
#include "../include/io.h"
#include "../include/signals.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>
#include <ftxui/screen/string.hpp>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int          kSessionRailWidth  = 30;
constexpr int          kInputChromeWidth  = kSessionRailWidth + 2;
constexpr int          kPathModalWidth    = 72;
constexpr const char * kPromptPlaceholder = "Paste text here, or type / for commands";

struct WrappedGlyph {
    std::string text;
    size_t      position    = 0;
    int         width       = 0;
    bool        break_after = false;
};

struct VisualLine {
    std::vector<WrappedGlyph> glyphs;
    int                       width = 0;
};

ftxui::Color accent_color() {
    return ftxui::Color::RGB(116, 238, 170);
}

ftxui::Color violet_color() {
    return ftxui::Color::RGB(177, 151, 252);
}

ftxui::ButtonOption quiet_button() {
    ftxui::ButtonOption option;
    option.transform = [](const ftxui::EntryState & state) {
        using namespace ftxui;

        auto label = text(state.label);
        if (state.focused) {
            label |= bold;
            label |= underlined;
            label |= color(accent_color());
        } else {
            label |= color(Color::GrayLight);
        }
        return hbox({
            text(state.focused ? "› " : "  ") | color(accent_color()),
            label,
            text("  "),
        });
    };
    return option;
}

ftxui::Element key_hint(const std::string & key, const std::string & action) {
    using namespace ftxui;
    return hbox({
        text(key) | bold | color(accent_color()),
        text(" " + action) | dim,
    });
}

ftxui::Element metric(const std::string & label, const std::string & value, const bool primary = false) {
    using namespace ftxui;
    return hbox({
        text(label) | dim,
        filler(),
        text(value) | bold | color(primary ? violet_color() : Color::White),
    });
}

ftxui::Element style_input(ftxui::InputState state) {
    using namespace ftxui;
    state.element |= color(Color::White);
    if (state.is_placeholder) {
        state.element |= dim;
    }
    if (state.focused) {
        state.element |= bold;
    }
    return state.element;
}

std::vector<WrappedGlyph> glyphs_from_line(const std::string & line, const size_t source_offset) {
    using namespace ftxui;

    std::vector<WrappedGlyph> glyphs;
    size_t                    position = source_offset;
    for (const auto & glyph : Utf8ToGlyphs(line)) {
        if (glyph.empty()) {
            continue;
        }
        glyphs.push_back({
            glyph,
            position,
            std::max(string_width(glyph), 0),
            glyph == " ",
        });
        position += glyph.size();
    }
    return glyphs;
}

// Wrap at the last space that fits. A single overlong word falls back to
// character wrapping so it can never overflow the input column.
std::vector<VisualLine> wrap_glyphs(const std::vector<WrappedGlyph> & glyphs, const int available_width) {
    const int               wrap_width = std::max(available_width, 1);
    std::vector<VisualLine> lines;
    size_t                  line_start = 0;

    while (line_start < glyphs.size()) {
        size_t line_end   = line_start;
        size_t last_break = line_start;
        int    line_width = 0;

        while (line_end < glyphs.size()) {
            const int next_width = line_width + glyphs[line_end].width;
            if (next_width > wrap_width && line_end > line_start) {
                break;
            }
            line_width = next_width;
            if (glyphs[line_end].break_after) {
                last_break = line_end + 1;
            }
            ++line_end;
        }

        if (line_end < glyphs.size() && last_break > line_start) {
            line_end   = last_break;
            line_width = 0;
            for (size_t index = line_start; index < line_end; ++index) {
                line_width += glyphs[index].width;
            }
        }

        VisualLine line;
        line.glyphs.assign(glyphs.begin() + static_cast<std::ptrdiff_t>(line_start),
                           glyphs.begin() + static_cast<std::ptrdiff_t>(line_end));
        line.width = line_width;
        lines.push_back(std::move(line));
        line_start = line_end;
    }

    if (lines.empty()) {
        lines.emplace_back();
    }
    return lines;
}

ftxui::Element render_visual_line(const VisualLine & line, const int cursor_position, const bool focused) {
    using namespace ftxui;

    Elements elements;
    for (const auto & glyph : line.glyphs) {
        auto element = text(glyph.text);
        if (cursor_position == static_cast<int>(glyph.position)) {
            element = focused ? focusCursorBarBlinking(std::move(element)) : ftxui::select(std::move(element));
        }
        elements.push_back(std::move(element));
    }
    if (elements.empty()) {
        elements.push_back(text(""));
    }
    return hbox(std::move(elements));
}

void append_cursor(std::vector<VisualLine> & lines, const size_t source_position, const int available_width) {
    WrappedGlyph cursor{ " ", source_position, 1, false };
    if (lines.back().width < std::max(available_width, 1)) {
        lines.back().glyphs.push_back(std::move(cursor));
        ++lines.back().width;
        return;
    }
    lines.push_back({ { std::move(cursor) }, 1 });
}

// This is a visual-only wrap. Byte positions still refer to the original
// string, so editing and analysis receive exactly the text the user entered.
ftxui::Element soft_wrapped_text(const std::string & content,
                                 int                 cursor_position,
                                 const bool          focused,
                                 const int           available_width) {
    using namespace ftxui;

    if (content.empty()) {
        auto placeholder = text(kPromptPlaceholder) | dim;
        if (focused) {
            placeholder |= focus;
        }
        return placeholder;
    }

    cursor_position = std::clamp(cursor_position, 0, static_cast<int>(content.size()));

    Elements lines;
    size_t   line_start = 0;
    while (line_start <= content.size()) {
        const size_t line_end = content.find('\n', line_start);
        const size_t end      = line_end == std::string::npos ? content.size() : line_end;

        auto visual_lines =
            wrap_glyphs(glyphs_from_line(content.substr(line_start, end - line_start), line_start), available_width);
        if (cursor_position == static_cast<int>(end)) {
            append_cursor(visual_lines, end, available_width);
        }
        for (const auto & visual_line : visual_lines) {
            lines.push_back(render_visual_line(visual_line, cursor_position, focused));
        }

        if (line_end == std::string::npos) {
            break;
        }
        line_start = line_end + 1;
    }

    auto result = vbox(std::move(lines)) | color(Color::White);
    if (focused) {
        result |= bold;
    }
    return result;
}

ftxui::Elements slash_command_rows(const std::vector<std::string> & commands, const int selected_index) {
    using namespace ftxui;

    Elements rows;
    for (int index = 0; index < static_cast<int>(commands.size()); ++index) {
        const bool selected    = index == selected_index;
        auto       command     = text(commands[index]) | bold;
        auto       description = text("  " + command_description(commands[index]));
        if (selected) {
            command |= color(accent_color());
            description |= color(Color::White);
        } else {
            command |= dim;
            description |= dim;
        }
        rows.push_back(hbox({
            text(selected ? "› " : "  ") | color(accent_color()),
            command,
            description,
            text("    "),
        }));
    }
    return rows;
}

ftxui::Element session_rail(const BackendSnapshot & snapshot, const bool model_busy, const int animation_frame) {
    using namespace ftxui;

    const auto indicator = model_busy ? spinner(6, static_cast<std::size_t>(animation_frame)) | color(accent_color()) :
                                        text(snapshot.model_ready ? "✓" : "–") |
                                            color(snapshot.model_ready ? accent_color() : Color::GrayDark);
    const std::string model_name  = model_label();
    const std::string model_state = snapshot.model_ready ? "ready" : (model_busy ? "working" : "not loaded");

    return vbox({
               text("SESSION") | bold | color(violet_color()),
               text(""),
               paragraph(snapshot.interpretation) | bold,
               text(snapshot.input_source) | dim,
               text(""),
               metric("direction", snapshot.score_direction, true),
               metric("classification", snapshot.classification, true),
               separator(),
               metric("score", snapshot.score_text),
               metric("threshold", snapshot.threshold_text),
               metric("tokens", snapshot.token_count),
               metric("speed", snapshot.speed),
               metric("time", snapshot.elapsed),
               filler(),
               text("MODEL") | bold | color(accent_color()),
               text(""),
               hbox({
                   indicator,
                   text(" "),
                   text(model_name) | bold,
               }),
               text(model_state) | dim,
               text(""),
               paragraph(snapshot.operation_status) | color(accent_color()),
           }) |
           size(WIDTH, EQUAL, kSessionRailWidth);
}

ftxui::Element path_picker_view(ftxui::Element path_input, const std::string & message) {
    using namespace ftxui;

    auto feedback = message.empty() ? text("Drag a file here, paste its path, or type it directly.") | dim :
                                      paragraph(message) | color(Color::RedLight);
    return vbox({
               hbox({
                   text("◆ OPEN FILE") | bold | color(accent_color()),
                   filler(),
                   text(".txt  .md") | dim,
               }),
               text(""),
               hbox({
                   text("› ") | bold | color(accent_color()),
                   std::move(path_input) | flex,
               }),
               separator(),
               feedback,
               text(""),
               hbox({
                   key_hint("enter", "insert into input"),
                   text("    "),
                   key_hint("esc", "close"),
               }),
           }) |
           borderRounded | bgcolor(Color::Grey3) | size(WIDTH, EQUAL, kPathModalWidth);
}

ftxui::Element main_view(const BackendSnapshot &          snapshot,
                         ftxui::Element                   input,
                         ftxui::Element                   footer,
                         const std::vector<std::string> & commands,
                         const int                        command_index,
                         const bool                       model_busy,
                         const int                        animation_frame,
                         const bool                       modal_open) {
    using namespace ftxui;

    Elements input_section = {
        std::move(input) | vscroll_indicator | yframe | flex,
    };
    if (!commands.empty()) {
        input_section.push_back(separator());
        input_section.push_back(hbox(slash_command_rows(commands, command_index)));
    }

    auto document = vbox({
                        hbox({
                            text("◆") | bold | color(accent_color()),
                            text(" DETECT") | bold,
                            text("LLAMA") | bold | color(accent_color()),
                        }),
                        separator(),
                        hbox({
                            vbox(std::move(input_section)) | flex,
                            separator(),
                            session_rail(snapshot, model_busy, animation_frame),
                        }) | flex,
                        separator(),
                        std::move(footer),
                    }) |
                    flex;
    if (modal_open) {
        document |= dim;
    }
    return document;
}

void join_if_running(std::thread & worker) {
    if (worker.joinable()) {
        worker.join();
    }
}

}  // namespace

int run_tui(const AppConfig & config) {
    using namespace ftxui;

    BackendSession   backend(config);
    std::thread      model_worker;
    std::thread      analysis_worker;
    std::thread      ticker;
    std::atomic_bool keep_ticking = true;

    std::string      prompt;
    std::string      path_value;
    std::string      path_message;
    std::atomic_bool model_busy        = true;
    std::atomic_bool analysis_busy     = false;
    bool             slash_menu_open   = false;
    bool             path_modal_open   = false;
    int              slash_menu_index  = 0;
    std::atomic_int  animation_frame   = 0;

    auto screen = ScreenInteractive::Fullscreen();
    auto redraw = [&] {
        screen.PostEvent(Event::Custom);
    };

    auto busy = [&] {
        return model_busy.load() || analysis_busy.load();
    };

    auto refresh_slash_menu_state = [&] {
        const auto matches = slash_command_matches(prompt);
        slash_menu_open    = !path_modal_open && !matches.empty();
        if (slash_menu_open) {
            slash_menu_index = std::clamp(slash_menu_index, 0, static_cast<int>(matches.size()) - 1);
        } else {
            slash_menu_index = 0;
        }
    };

    // A thread object is reused for model operations, so the completed worker
    // is joined before a new load or download starts.
    auto start_model_task = [&](std::function<void()> task) {
        join_if_running(model_worker);
        model_busy   = true;
        model_worker = std::thread([&, task = std::move(task)] {
            task();
            model_busy = false;
            redraw();
        });
    };

    auto download_fixed_model = [&] {
        if (busy()) {
            backend.set_operation_status("Please wait for the current operation to finish.");
            return;
        }
        slash_menu_open = false;
        prompt.clear();
        redraw();
        start_model_task([&] { backend.download_and_load_model(); });
    };

    auto start_analysis = [&](const DetectionInput & input_value) {
        if (busy()) {
            backend.set_operation_status("Please wait for the current operation to finish.");
            return false;
        }
        if (!backend.snapshot().model_ready) {
            backend.set_operation_status("Model is not ready yet.");
            return false;
        }

        analysis_busy = true;

        join_if_running(analysis_worker);
        analysis_worker = std::thread([&, input_value] {
            backend.analyze_input(input_value);
            analysis_busy = false;
            redraw();
        });
        return true;
    };

    int         prompt_cursor_position = 0;
    InputOption input_options;
    input_options.content         = &prompt;
    input_options.multiline       = true;
    input_options.cursor_position = &prompt_cursor_position;
    input_options.on_change       = [&] {
        refresh_slash_menu_state();
    };
    auto input        = Input(input_options);
    auto focus_prompt = [&] {
        input->TakeFocus();
        redraw();
    };
    Box  input_box;
    auto wrapped_input = Renderer(input, [&] {
        const int available_width = screen.dimx() - kInputChromeWidth;
        return soft_wrapped_text(prompt, prompt_cursor_position, input->Focused(), available_width) |
               reflect(input_box);
    });
    wrapped_input |= CatchEvent([&](Event event) {
        if (!event.is_mouse() || event.mouse().button != Mouse::Left || event.mouse().motion != Mouse::Pressed ||
            !input_box.Contain(event.mouse().x, event.mouse().y)) {
            return false;
        }
        input->TakeFocus();
        redraw();
        return true;
    });

    std::function<void()> submit_path;
    InputOption           path_input_options;
    path_input_options.content     = &path_value;
    path_input_options.placeholder = "./notes/sample.md";
    path_input_options.on_change   = [&] {
        path_message.clear();
    };
    path_input_options.on_enter = [&] {
        submit_path();
    };
    path_input_options.transform = style_input;
    auto path_input              = Input(path_input_options);

    auto close_path_modal = [&] {
        path_modal_open = false;
        path_message.clear();
        focus_prompt();
    };

    auto open_path_modal = [&] {
        path_value = slash_command_argument(prompt, "/path");
        path_message.clear();
        path_modal_open   = true;
        slash_menu_open   = false;
        prompt.clear();
        prompt_cursor_position = 0;
        backend.set_operation_status("Choose a local .txt or .md file.");
        path_input->TakeFocus();
        redraw();
    };

    // /path is an editor import: preserve the file contents in the prompt and
    // leave inference under the explicit Analyze action.
    auto load_file_into_prompt = [&](const DetectionInput & input_value) {
        namespace fs = std::filesystem;

        const fs::path path(input_value.value);
        std::error_code path_error;
        if (!fs::exists(path, path_error) || !fs::is_regular_file(path, path_error)) {
            path_message = "Input must be an existing regular file.";
        } else if (!is_supported_input_file(path)) {
            path_message = "Only .md and .txt files are supported.";
        } else {
            std::string contents;
            if (!read_file_to_string(path.string(), contents)) {
                path_message = "Failed to read the selected file.";
            } else {
                prompt                  = std::move(contents);
                prompt_cursor_position  = static_cast<int>(prompt.size());
                path_modal_open         = false;
                slash_menu_open         = false;
                path_message.clear();
                backend.set_operation_status("Loaded " + path.filename().string() +
                                             " into the input. Analysis has not started.");
                refresh_slash_menu_state();
                focus_prompt();
                return true;
            }
        }

        path_value        = input_value.value;
        path_modal_open   = true;
        slash_menu_open   = false;
        backend.set_operation_status(path_message);
        path_input->TakeFocus();
        redraw();
        return false;
    };

    submit_path = [&] {
        const PromptParseResult parsed = parse_prompt_input("/path " + path_value);
        if (parsed.action != PromptAction::LoadFile || parsed.input.kind != DetectionInputKind::File) {
            path_message = parsed.message.empty() ? "Enter a valid .txt or .md file path." : parsed.message;
            redraw();
            return;
        }
        load_file_into_prompt(parsed.input);
    };

    auto apply_slash_command = [&](const std::string & command) {
        if (command == "/download") {
            download_fixed_model();
            return;
        }
        if (command == "/path") {
            open_path_modal();
        }
    };

    auto analyze = [&] {
        const std::string raw_prompt = prompt;
        if (trim_copy(raw_prompt) == "/path") {
            open_path_modal();
            return;
        }

        const PromptParseResult parsed = parse_prompt_input(raw_prompt);
        switch (parsed.action) {
            case PromptAction::Empty:
            case PromptAction::UnknownCommand:
                backend.set_operation_status(parsed.message);
                return;
            case PromptAction::DownloadModel:
                download_fixed_model();
                return;
            case PromptAction::LoadFile:
                load_file_into_prompt(parsed.input);
                return;
            case PromptAction::Analyze:
                start_analysis(parsed.input);
                return;
        }
    };

    auto clear = [&] {
        if (busy()) {
            backend.set_operation_status("Cannot reset while an operation is running.");
            return;
        }
        prompt.clear();
        path_value.clear();
        path_message.clear();
        slash_menu_open   = false;
        path_modal_open   = false;
        backend.clear_analysis();
        focus_prompt();
    };

    auto quit = [&] {
        if (busy()) {
            backend.set_operation_status("An operation is running; wait for it to finish before quitting.");
            return;
        }
        screen.ExitLoopClosure()();
    };

    auto analyze_button = Button("analyze", analyze, quiet_button());
    auto clear_button   = Button("reset", clear, quiet_button());
    auto quit_button    = Button("quit", quit, quiet_button());

    auto buttons = Container::Horizontal({
        analyze_button,
        clear_button,
        quit_button,
    });

    auto main_container = Container::Vertical({
        wrapped_input,
        buttons,
    });

    auto main_renderer = Renderer(main_container, [&] {
        const BackendSnapshot snapshot = backend.snapshot();
        const auto            commands = slash_menu_open ? slash_command_matches(prompt) : std::vector<std::string>{};
        return main_view(snapshot, wrapped_input->Render(), buttons->Render(), commands, slash_menu_index,
                         model_busy.load(), animation_frame.load(), path_modal_open);
    });

    main_renderer |= CatchEvent([&](Event event) {
        if (!slash_menu_open) {
            return false;
        }

        const auto matches = slash_command_matches(prompt);
        if (matches.empty()) {
            return false;
        }

        if (event == Event::Escape) {
            slash_menu_open = false;
            backend.set_operation_status("Command menu closed.");
            redraw();
            return true;
        }
        if (event == Event::ArrowDown || event == Event::Tab) {
            slash_menu_index = (slash_menu_index + 1) % static_cast<int>(matches.size());
            redraw();
            return true;
        }
        if (event == Event::ArrowUp || event == Event::TabReverse) {
            slash_menu_index =
                (slash_menu_index + static_cast<int>(matches.size()) - 1) % static_cast<int>(matches.size());
            redraw();
            return true;
        }
        if (event == Event::Return) {
            const int index = std::clamp(slash_menu_index, 0, static_cast<int>(matches.size()) - 1);
            apply_slash_command(matches[index]);
            return true;
        }
        return false;
    });

    auto path_modal = Renderer(path_input, [&] { return path_picker_view(path_input->Render(), path_message); });

    path_modal |= CatchEvent([&](Event event) {
        if (event == Event::Escape) {
            close_path_modal();
            backend.set_operation_status("File picker closed.");
            return true;
        }
        return false;
    });

    Component application = main_renderer;
    application |= Modal(path_modal, &path_modal_open);

    // FTXUI has no timer events, so a lightweight worker advances the model
    // spinner and forwards process interrupts to the UI loop.
    ticker = std::thread([&] {
        while (keep_ticking) {
            if (g_interrupted) {
                keep_ticking = false;
                backend.set_operation_status("Interrupt requested. Exiting DetectLlama.");
                screen.ExitLoopClosure()();
                break;
            }
            const bool should_animate = model_busy.load();
            if (should_animate) {
                ++animation_frame;
                redraw();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
    });

    start_model_task([&] { backend.initialize(); });
    screen.Loop(application);

    keep_ticking = false;
    join_if_running(ticker);
    join_if_running(model_worker);
    join_if_running(analysis_worker);
    return 0;
}
