#pragma once
#include "llama.h"

#include <atomic>
#include <filesystem>
#include <string>

inline std::atomic<bool> g_interrupted(false);

struct LlamaState {
    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context *     ctx   = nullptr;
};

bool setup_llama(LlamaState & llama, const std::string & model_path, bool gpu, int n_ctx, int n_batch);

// Custom logging callback that only print errors
void custom_log(ggml_log_level level, const char * text, void * user_data);

void print_logo();

void signal_handler(const int signum);
