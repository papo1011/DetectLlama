#include "../include/model_catalog.h"

namespace {

constexpr const char * LLAMA3_8B_REPO = "QuantFactory/Meta-Llama-3-8B-GGUF";
constexpr const char * LLAMA3_8B_F16_REPO = "Orneyfish/Meta-Llama-3-8B_F_16.gguf";
constexpr const char * CALIBRATION_ID = "ghostbuster-953b44b7-seed20260730-ctx512";
constexpr int          CALIBRATION_CONTEXT = 512;

}  // namespace

const std::vector<ModelInfo> & model_catalog() {
    static const std::vector<ModelInfo> models = {
        {"Llama 3 8B", "Q4_K_S", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_K_S.gguf", 4694, 10, -2.0881,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q4_0", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_0.gguf", 4662, 10, -2.041442,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q4_K_M", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_K_M.gguf", 4922, 11, -2.2471,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q4_1", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q4_1.gguf", 5131, 11, -2.5062,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q5_K_S", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_K_S.gguf", 5600, 12, -2.6293,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q5_0", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_0.gguf", 5600, 12, -2.4060,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q5_K_M", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_K_M.gguf", 5734, 13, -2.4852,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q5_1", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q5_1.gguf", 6069, 13, -2.7678,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q6_K", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q6_K.gguf", 6597, 14, -2.4378,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "Q8_0", LLAMA3_8B_REPO, "Meta-Llama-3-8B.Q8_0.gguf", 8542, 15, -2.5163,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
        {"Llama 3 8B", "FP16", LLAMA3_8B_F16_REPO, "Meta-Llama-3-8B_F_16.gguf", 16070, 16, -2.4794,
         CALIBRATION_CONTEXT, CALIBRATION_ID},
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

const ModelInfo * find_model_by_filename(const std::string & filename) {
    for (const auto & model : model_catalog()) {
        if (model.filename == filename) {
            return &model;
        }
    }
    return nullptr;
}
