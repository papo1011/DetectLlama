#include "../include/app_config.h"
#include "../include/backend.h"
#include "../include/llama_logging.h"
#include "../include/signals.h"

#include <argparse/argparse.hpp>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

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

std::string read_stdin() {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    return buffer.str();
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

void print_json_result(const BackendSnapshot & snapshot, const DetectionInput & input, const AnalysisResult & result) {
    const double probability = result.ok ? ai_probability_from_score(result.discrepancy) : 0.0;
    std::cout << std::boolalpha;
    std::cout << "{\n";
    std::cout << "  \"ok\": " << result.ok << ",\n";
    std::cout << "  \"input\": {\n";
    std::cout << "    \"kind\": \"" << (input.kind == DetectionInputKind::File ? "file" : "text") << "\",\n";
    std::cout << "    \"source\": \"" << json_escape(input.source_label) << "\"\n";
    std::cout << "  },\n";
    std::cout << "  \"model\": {\n";
    std::cout << "    \"path\": \"" << json_escape(snapshot.loaded_model_path) << "\",\n";
    std::cout << "    \"label\": \"" << json_escape(snapshot.loaded_model_quant) << "\"\n";
    std::cout << "  },\n";
    std::cout << "  \"score\": " << result.discrepancy << ",\n";
    std::cout << "  \"ai_probability\": " << probability << ",\n";
    std::cout << "  \"interpretation\": \"" << json_escape(result.ok ? interpret_score(result.discrepancy) : snapshot.interpretation)
              << "\",\n";
    std::cout << "  \"tokens\": " << result.tokens << ",\n";
    std::cout << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n";
    std::cout << "  \"tokens_per_second\": " << result.tokens_per_second << ",\n";
    std::cout << "  \"error\": \"" << json_escape(result.error) << "\"\n";
    std::cout << "}\n";
}

void print_human_result(const BackendSnapshot & snapshot, const AnalysisResult & result) {
    if (!result.ok) {
        std::cerr << result.error << "\n";
        return;
    }

    std::cout << "Model: " << snapshot.loaded_model_quant << " (" << snapshot.loaded_model_path << ")\n";
    std::cout << "Score: " << format_fixed(result.discrepancy, 4) << "\n";
    std::cout << "AI probability: " << format_percent(ai_probability_from_score(result.discrepancy)) << "\n";
    std::cout << "Interpretation: " << interpret_score(result.discrepancy) << "\n";
    std::cout << "Tokens: " << result.tokens << "\n";
    std::cout << "Elapsed: " << format_fixed(result.elapsed_seconds, 2) << " s\n";
    std::cout << "Speed: " << format_fixed(result.tokens_per_second, 2) << " tokens/sec\n";
}

}  // namespace

int main(const int argc, char * argv[]) {
    clear_hf_token_environment();
    install_signal_handlers();

    argparse::ArgumentParser program("DetectLlamaBackend", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);
    program.add_argument("--json").help("Print a JSON result").default_value(false).implicit_value(true);
    program.add_argument("--stdin").help("Read input text from stdin").default_value(false).implicit_value(true);
    program.add_argument("--model-path").help("Local GGUF model path").default_value(std::string(""));
    program.add_argument("--model-label").help("Label to report for --model-path").default_value(std::string("local"));
    program.add_argument("--text").help("Inline text to analyze").default_value(std::string(""));
    program.add_argument("--file").help("Path to a .txt or .md file to analyze").default_value(std::string(""));
    program.add_argument("-c", "--ctx").help("Size of the prompt context").default_value(2048).scan<'i', int>();
    program.add_argument("-b", "--batch").help("Logical max batch size").default_value(2048).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception & err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const bool verbose = program.get<bool>("--verbose");
    if (!verbose) {
        llama_log_set(custom_log, nullptr);
    }

    const std::string model_path = program.get<std::string>("--model-path");
    if (model_path.empty()) {
        std::cerr << "--model-path is required for headless backend analysis.\n";
        return 1;
    }

    const std::string inline_text = program.get<std::string>("--text");
    const std::string file_path = program.get<std::string>("--file");
    const bool        use_stdin = program.get<bool>("--stdin");
    const int         input_count = (inline_text.empty() ? 0 : 1) + (file_path.empty() ? 0 : 1) + (use_stdin ? 1 : 0);
    if (input_count != 1) {
        std::cerr << "Provide exactly one input: --text, --file, or --stdin.\n";
        return 1;
    }

    DetectionInput input;
    if (!file_path.empty()) {
        input.kind = DetectionInputKind::File;
        input.value = normalize_dropped_path(file_path);
        input.source_label = "File: " + std::filesystem::path(input.value).filename().string();
    } else {
        input.kind = DetectionInputKind::Text;
        input.value = use_stdin ? trim_copy(read_stdin()) : inline_text;
        input.source_label = "Pasted text";
    }

    AppConfig config;
    config.use_gpu = program.get<bool>("--gpu");
    config.n_ctx = program.get<int>("--ctx");
    config.n_batch = program.get<int>("--batch");

    BackendSession backend(config);
    if (!backend.load_model_path(model_path, program.get<std::string>("--model-label"))) {
        const auto snapshot = backend.snapshot();
        if (program.get<bool>("--json")) {
            AnalysisResult result;
            result.error = snapshot.operation_status;
            print_json_result(snapshot, input, result);
        } else {
            std::cerr << snapshot.operation_status << "\n";
        }
        return 2;
    }

    const AnalysisResult result = backend.analyze_input(input);
    const BackendSnapshot snapshot = backend.snapshot();

    if (program.get<bool>("--json")) {
        print_json_result(snapshot, input, result);
    } else {
        print_human_result(snapshot, result);
    }

    return result.ok ? 0 : 3;
}
