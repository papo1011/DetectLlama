#include "../include/model_catalog.h"

const std::vector<ModelInfo> & model_catalog() {
    static const std::vector<ModelInfo> models = {
        {"Q4_0", "ggml-tiiuae-falcon-7b-instruct-Q4_0.gguf", 3900, 1},
        {"Q4_K_M", "tiiuae-falcon-7b-instruct-Q4_K_M.gguf", 4100, 2},
        {"Q5_K_M", "tiiuae-falcon-7b-instruct-Q5_K_M.gguf", 5100, 3},
        {"Q6_K", "tiiuae-falcon-7b-instruct-Q6_K.gguf", 5900, 4},
        {"Q8_0", "ggml-tiiuae-falcon-7b-instruct-Q8_0.gguf", 7600, 5},
    };
    return models;
}

const ModelInfo * find_model_by_quant(const std::string & quant) {
    for (const auto & model : model_catalog()) {
        if (model.quant == quant) {
            return &model;
        }
    }
    return nullptr;
}

const ModelInfo * find_model_by_rank(const int rank) {
    for (const auto & model : model_catalog()) {
        if (model.rank == rank) {
            return &model;
        }
    }
    return nullptr;
}
