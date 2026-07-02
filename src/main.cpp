#include "../include/detect.h"
#include "../include/io.h"
#include "../include/utils.h"

#include <argparse/argparse.hpp>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(const int argc, char * argv[]) {
    print_logo();

    std::signal(SIGINT, signal_handler);

    argparse::ArgumentParser program("fast-detect-gpt", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);
    program.add_argument("-m", "--model")
        .help("Path to the GGUF model file")
        .default_value("../models/tiiuae-falcon-7b-instruct-Q5_K_M.gguf");
    program.add_argument("-f", "--file").help("Path to the input text file").required();
    program.add_argument("-c", "--ctx").help("Size of the prompt context").default_value(4096).scan<'i', int>();
    program.add_argument("-b", "--batch").help("Logical max batch size").default_value(4096).scan<'i', int>();

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
    const auto   input_file  = program.get<std::string>("--file");
    const int    n_ctx       = program.get<int>("--ctx");
    const int    n_batch     = program.get<int>("--batch");

    if (!std::filesystem::exists(input_file) || !std::filesystem::is_regular_file(input_file)) {
        std::cerr << "Input must be an existing regular file: " << input_file << std::endl;
        return 1;
    }

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

    std::cout << "Processing text file: " << input_file << std::endl;

    if (std::string input_text; !read_file_to_string(input_file, input_text)) {
        std::cerr << "Failed to read input file." << std::endl;
    } else {
        const double discrepancy = analyze_text(llama, input_text, n_ctx);
        std::cout << "DISCREPANCY: " << std::fixed << std::setprecision(4) << discrepancy << std::endl;
    }

    llama_free(llama.ctx);
    llama_model_free(llama.model);
    llama_backend_free();

    return 0;
}
