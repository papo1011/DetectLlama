#include "../include/app_config.h"
#include "../include/llama_logging.h"
#include "../include/signals.h"
#include "../include/tui.h"

#include <argparse/argparse.hpp>
#include <array>
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
    install_signal_handlers();

    argparse::ArgumentParser program("DetectLlama", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);

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

    return run_tui(config);
}
