#include "../include/llama_state.h"

bool setup_llama(LlamaState & llama, const std::string & model_path, bool gpu, int n_ctx, int n_batch) {
    auto mparams = llama_model_default_params();

    if (gpu) {
        mparams.n_gpu_layers = 200;
    } else {
        mparams.n_gpu_layers = 0;
    }

    llama.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!llama.model) {
        return false;
    }

    llama.vocab = llama_model_get_vocab(llama.model);

    auto cparams       = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_batch;
    cparams.embeddings = true;

    llama.ctx = llama_init_from_model(llama.model, cparams);
    return (llama.ctx != nullptr);
}

void free_llama_state(LlamaState & llama) {
    if (llama.ctx) {
        llama_free(llama.ctx);
        llama.ctx = nullptr;
    }
    if (llama.model) {
        llama_model_free(llama.model);
        llama.model = nullptr;
    }
    llama.vocab = nullptr;
}
