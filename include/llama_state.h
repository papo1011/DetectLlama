#pragma once

#include "llama.h"

#include <string>

struct LlamaState {
    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context *     ctx   = nullptr;
};

bool setup_llama(LlamaState & llama, const std::string & model_path, bool gpu, int n_ctx, int n_batch);
void free_llama_state(LlamaState & llama);
