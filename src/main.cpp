#include "../include/app_config.h"
#include "../include/llama_logging.h"
#include "../include/signals.h"
#include "../include/tui.h"

#include <argparse/argparse.hpp>
#include <csignal>
#include <filesystem>
#include <iostream>

int main(const int argc, char * argv[]) {
    std::signal(SIGINT, signal_handler);

    argparse::ArgumentParser program("DetectLlama", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);
    program.add_argument("--model-repo")
        .help("Hugging Face GGUF repository")
        .default_value(std::string("maddes8cht/tiiuae-falcon-7b-instruct-gguf"));
    program.add_argument("--llama-cli").help("Path to llama-cli").default_value(std::string(""));
    program.add_argument("--build-dir").help("Build directory used to find bundled llama-cli").default_value(std::string(""));
    program.add_argument("--target-tps").help("Target tokens/sec for automatic quant selection").default_value(30).scan<'i', int>();
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

    AppConfig config;
    config.use_gpu = program.get<bool>("--gpu");
    config.model_repo = program.get<std::string>("--model-repo");
    config.llama_cli = program.get<std::string>("--llama-cli");
    config.build_dir = program.get<std::string>("--build-dir");
    config.target_tokens_per_sec = program.get<int>("--target-tps");
    config.n_ctx = program.get<int>("--ctx");
    config.n_batch = program.get<int>("--batch");

    if (config.build_dir.empty()) {
        std::error_code error;
        config.build_dir = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path(), error).string();
    }

    return run_tui(config);
}
