#pragma once

#include <string>

struct AppConfig {
    std::string model_repo = "maddes8cht/tiiuae-falcon-7b-instruct-gguf";
    std::string build_dir;
    std::string llama_cli;
    int         n_ctx = 2048;
    int         n_batch = 2048;
    int         target_tokens_per_sec = 30;
    bool        use_gpu = false;
};
