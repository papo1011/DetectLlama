#include "../include/detect.h"
#include "../include/io.h"
#include "../include/utils.h"

#include <argparse/argparse.hpp>
#include <atomic>
#include <csignal>

int main(const int argc, char * argv[]) {
    print_logo();

    std::signal(SIGINT, signal_handler);

    argparse::ArgumentParser program("fast-detect-gpt", "0.1.0");

    program.add_argument("--verbose").help("Verbosity level").default_value(false).implicit_value(true);
    program.add_argument("--gpu").help("Enable GPU acceleration").default_value(false).implicit_value(true);
    program.add_argument("-m", "--model")
        .help("Path to the GGUF model file")
        .default_value("../models/tiiuae-falcon-7b-instruct-Q5_K_M.gguf");
    program.add_argument("-f", "--file").help("Path to the input file (txt or parquet)").required();
    program.add_argument("-c", "--ctx").help("Size of the prompt context").default_value(4096).scan<'i', int>();
    program.add_argument("-b", "--batch").help("Logical max batch size").default_value(4096).scan<'i', int>();
    program.add_argument("--col").help("Column name to analyze, Parquet only").default_value(std::string("text"));
    program.add_argument("-o", "--output")
        .help("Output file path (Parquet only)")
        .default_value(std::string("output_scored.parquet"));

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
    const auto   col_name    = program.get<std::string>("--col");
    const auto   output_file = program.get<std::string>("--output");
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

    if (input_file.ends_with(".parquet")) {
        std::cout << "Detected Parquet file. Reading column: '" << col_name << "'" << std::endl;
        auto [table, texts] = load_parquet_and_get_text(input_file, col_name);

        if (!table || texts.empty()) {
            std::cerr << "Failed to load Parquet or column is empty." << std::endl;
            llama_free(llama.ctx);
            llama_model_free(llama.model);
            llama_backend_free();
            return 1;
        }

        std::cout << "Loaded " << texts.size() << " rows. Inference started!" << std::endl;
        std::vector<double> scores;
        scores.reserve(texts.size());

        for (size_t i = 0; i < texts.size(); ++i) {
            if (g_interrupted) {
                std::cout << "\nProcess interrupted by user at row " << i << std::endl;
                break;
            }

            std::cout << "--------------------------------" << std::endl;
            std::cout << "Processing row " << i + 1 << std::endl;

            double score = analyze_text(llama, texts[i], n_ctx);
            std::cout << "DISCREPANCY: " << score << std::endl;
            scores.push_back(score);
        }
        std::cout << std::endl;

        if (!scores.empty()) {
            std::cout << "Saving " << scores.size() << " results to: " << output_file << std::endl;
            std::shared_ptr<arrow::Table> table_to_save = table;

            if (scores.size() < static_cast<size_t>(table->num_rows())) {
                std::cout << "Warning: Saving partial results (" << scores.size() << " out of " << table->num_rows()
                          << " rows)" << std::endl;
                // Slice(offset, length) create a view without save data
                table_to_save = table->Slice(0, scores.size());
            }

            if (save_parquet_with_scores(output_file, table_to_save, scores)) {
                std::cout << "Success! Saved." << std::endl;
            } else {
                std::cerr << "Failed to save output parquet file." << std::endl;
            }
        } else {
            std::cout << "No rows processed. Nothing to save." << std::endl;
        }

    } else {
        std::cout << "Processing single text file: " << input_file << std::endl;

        if (std::string input_text; !read_file_to_string(input_file, input_text)) {
            std::cerr << "Failed to read input file." << std::endl;
        } else {
            const double discrepancy = analyze_text(llama, input_text, n_ctx);
            std::cout << "DISCREPANCY: " << std::fixed << std::setprecision(4) << discrepancy << std::endl;
        }
    }

    llama_free(llama.ctx);
    llama_model_free(llama.model);
    llama_backend_free();

    return 0;
}
