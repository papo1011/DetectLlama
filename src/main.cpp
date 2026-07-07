#include "../include/app_config.h"
#include "../include/llama_logging.h"
#include "../include/signals.h"
#include "../include/tui.h"

#include <argparse/argparse.hpp>
#include <array>
#include <csignal>
#include <cstdlib>
#include <iostream>

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

}  // namespace

int main(const int argc, char * argv[]) {
    clear_hf_token_environment();
    std::signal(SIGINT, signal_handler);

    argparse::ArgumentParser program("DetectLlama", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);
    program.add_argument("--model-repo")
        .help("Hugging Face GGUF repository")
        .default_value(std::string("maddes8cht/tiiuae-falcon-7b-instruct-gguf"));
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
    config.target_tokens_per_sec = program.get<int>("--target-tps");
    config.n_ctx = program.get<int>("--ctx");
    config.n_batch = program.get<int>("--batch");

    return run_tui(config);
}
