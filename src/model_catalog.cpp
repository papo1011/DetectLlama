#include "../include/model_catalog.h"

namespace {

constexpr const char * LLAMA3_8B_REPO = "QuantFactory/Meta-Llama-3-8B-GGUF";
constexpr const char * LLAMA3_8B_F16_REPO = "Orneyfish/Meta-Llama-3-8B_F_16.gguf";
constexpr const char * LLAMA3_8B_INSTRUCT_REPO = "bartowski/Meta-Llama-3-8B-Instruct-GGUF";

}  // namespace

const std::vector<ModelInfo> & model_catalog() {
    static const std::vector<ModelInfo> models = {
        {"Llama 3 8B", "Q4_K_S", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_K_S.gguf", 4694, 10},
        {"Llama 3 8B", "Q4_0", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_0.gguf", 4662, 10},
        {"Llama 3 8B", "Q4_K_M", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_K_M.gguf", 4922, 11},
        {"Llama 3 8B", "Q4_1", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_1.gguf", 5131, 11},
        {"Llama 3 8B", "Q5_K_S", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_K_S.gguf", 5600, 12},
        {"Llama 3 8B", "Q5_0", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_0.gguf", 5600, 12},
        {"Llama 3 8B", "Q5_K_M", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_K_M.gguf", 5734, 13},
        {"Llama 3 8B", "Q5_1", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_1.gguf", 6069, 13},
        {"Llama 3 8B", "Q6_K", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q6_K.gguf", 6597, 14},
        {"Llama 3 8B", "Q8_0", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q8_0.gguf", 8542, 15},
        {"Llama 3 8B", "FP16", LLAMA3_8B_F16_REPO, "Meta-Llama-3-8B_F_16.gguf", 16070, 16},

        {"Llama 3 8B Instruct", "IQ4_XS", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-IQ4_XS.gguf", 4448, 9},
        {"Llama 3 8B Instruct", "IQ4_NL", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-IQ4_NL.gguf", 4678, 10},
        {"Llama 3 8B Instruct", "Q4_K_S", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-Q4_K_S.gguf", 4693, 10},
        {"Llama 3 8B Instruct", "Q4_K_M", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 4921, 11},
        {"Llama 3 8B Instruct", "Q5_K_S", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-Q5_K_S.gguf", 5600, 12},
        {"Llama 3 8B Instruct", "Q5_K_M", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf", 5733, 13},
        {"Llama 3 8B Instruct", "Q6_K", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-Q6_K.gguf", 6597, 14},
        {"Llama 3 8B Instruct", "Q8_0", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-Q8_0.gguf", 8541, 15},
        {"Llama 3 8B Instruct", "FP16", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-fp16.gguf", 16069, 16},
        {"Llama 3 8B Instruct", "FP32", LLAMA3_8B_INSTRUCT_REPO, "Meta-Llama-3-8B-Instruct-fp32.gguf", 32129, 17},
    };
    return models;
}

std::string model_label(const ModelInfo & model) {
    if (model.family.empty()) {
        return model.quant;
    }
    if (model.quant.empty()) {
        return model.family;
    }
    return model.family + " " + model.quant;
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
