#include "../include/tui.h"

#include "../include/detect.h"
#include "../include/io.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
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

}  // namespace

int run_tui(const LlamaState & llama, const int n_ctx) {
    namespace fs = std::filesystem;
    using namespace ftxui;

    std::mutex  state_mutex;
    std::thread worker;

    std::string input_path;
    std::string status = "Enter a local text file path, then run analysis.";
    std::string score_text = "-";
    std::string interpretation = "Waiting for input.";
    std::string token_count = "-";
    std::string elapsed = "-";
    std::string speed = "-";
    bool        busy = false;

    auto screen = ScreenInteractive::Fullscreen();
    auto input = Input(&input_path, "absolute/or/relative/path.txt");

    auto set_status = [&](std::string new_status) {
        std::lock_guard<std::mutex> lock(state_mutex);
        status = std::move(new_status);
    };

    auto analyze = [&] {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (busy) {
                status = "Analysis is already running.";
                return;
            }
            busy = true;
            status = "Analyzing file. Please wait...";
            score_text = "-";
            interpretation = "Running inference and scoring.";
            token_count = "-";
            elapsed = "-";
            speed = "measuring...";
        }

        const std::string path = input_path;
        if (worker.joinable()) {
            worker.join();
        }

        worker = std::thread([&, path] {
            if (!fs::exists(path) || !fs::is_regular_file(path)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    status = "Input must be an existing regular file.";
                    interpretation = "No analysis run.";
                    speed = "-";
                    busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            std::string input_text;
            if (!read_file_to_string(path, input_text)) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    status = "Failed to read input file.";
                    interpretation = "No analysis run.";
                    speed = "-";
                    busy = false;
                }
                screen.PostEvent(Event::Custom);
                return;
            }

            const AnalysisResult result = analyze_text_detailed(llama, input_text, n_ctx);
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                if (!result.ok) {
                    status = result.error;
                    interpretation = "No score produced.";
                    score_text = "-";
                    token_count = std::to_string(result.tokens);
                    elapsed = "-";
                    speed = "-";
                } else {
                    status = "Analysis complete.";
                    score_text = format_fixed(result.discrepancy, 4);
                    interpretation = interpret_score(result.discrepancy);
                    token_count = std::to_string(result.tokens);
                    elapsed = format_fixed(result.elapsed_seconds, 2) + " s";
                    speed = format_fixed(result.tokens_per_second, 2) + " tokens/sec";
                }
                busy = false;
            }
            screen.PostEvent(Event::Custom);
        });
    };

    auto clear = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (busy) {
            status = "Cannot clear while analysis is running.";
            return;
        }
        input_path.clear();
        status = "Enter a local text file path, then run analysis.";
        score_text = "-";
        interpretation = "Waiting for input.";
        token_count = "-";
        elapsed = "-";
        speed = "-";
    };

    auto quit = [&] {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (busy) {
            status = "Analysis is running; wait for it to finish before quitting.";
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
        std::string status_view;
        std::string score_view;
        std::string interpretation_view;
        std::string token_count_view;
        std::string elapsed_view;
        std::string speed_view;
        bool        busy_view;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            status_view = status;
            score_view = score_text;
            interpretation_view = interpretation;
            token_count_view = token_count;
            elapsed_view = elapsed;
            speed_view = speed;
            busy_view = busy;
        }

        auto status_line = busy_view ? text("Status: " + status_view) | color(Color::Yellow) | bold
                                     : text("Status: " + status_view);

        auto metrics = vbox({
                           hbox(text("Discrepancy") | bold, filler(), text(score_view)),
                           separator(),
                           hbox(text("Tokens"), filler(), text(token_count_view)),
                           hbox(text("Elapsed"), filler(), text(elapsed_view)),
                           hbox(text("Speed"), filler(), text(speed_view)),
                       }) |
                       border;

        auto body = hbox({
                        vbox({
                            text("File path") | bold,
                            input->Render() | border,
                            status_line,
                            separator(),
                            text("Result") | bold,
                            paragraph(interpretation_view),
                            filler(),
                            buttons->Render(),
                        }) | flex,
                        separator(),
                        metrics | size(WIDTH, GREATER_THAN, 30),
                    }) |
                    flex;

        return vbox({
                   hbox(text("DetectLlama") | bold, filler(), text("Fast-DetectGPT local text analysis") | dim),
                   separator(),
                   body,
                   separator(),
                   text("Tab/Shift+Tab moves focus. Enter activates buttons. Paste or type a full file path.") | dim,
               }) |
               border | flex;
    });

    screen.Loop(renderer);
    if (worker.joinable()) {
        worker.join();
    }
    return 0;
}
