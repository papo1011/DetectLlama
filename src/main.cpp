#include "../include/console.h"
#include "../include/detect.h"
#include "../include/io.h"
#include "../include/llama_logging.h"
#include "../include/llama_state.h"
#include "../include/signals.h"
#include "../include/tui.h"

#include <argparse/argparse.hpp>
#include <csignal>
#include <iostream>

int main(const int argc, char * argv[]) {
    print_logo();

    std::signal(SIGINT, signal_handler);

    argparse::ArgumentParser program("DetectLlama", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);
    program.add_argument("-m", "--model")
        .help("Path to the GGUF model file")
        .default_value("../models/tiiuae-falcon-7b-instruct-Q5_K_M.gguf");
    program.add_argument("-c", "--ctx").help("Size of the prompt context").default_value(2048).scan<'i', int>();
    program.add_argument("-b", "--batch").help("Logical max batch size").default_value(2048).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception & err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const bool   verbose     = program.get<bool>("--verbose");
    const bool   gpu         = program.get<bool>("--gpu");
    const auto   model_path  = program.get<std::string>("--model");
    const int    n_ctx       = program.get<int>("--ctx");
    const int    n_batch     = program.get<int>("--batch");

    if (!verbose) {
        llama_log_set(custom_log, nullptr);
    }

    std::cout << "Loading model..." << std::endl;

    llama_backend_init();

    LlamaState llama = {};
    if (!setup_llama(llama, model_path, gpu, n_ctx, n_batch)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }

    run_tui(llama, n_ctx);

    llama_free(llama.ctx);
    llama_model_free(llama.model);
    llama_backend_free();

    return 0;
}
