#pragma once

#include <string>
#include <vector>

struct ModelInfo {
    std::string family;
    std::string quant;
    std::string repo;
    std::string filename;
    int         size_mb = 0;
    int         rank = 0;
    double      threshold = 0.0;
    int         threshold_context = 0;
    std::string calibration_id;
};

const std::vector<ModelInfo> & model_catalog();
std::string                    model_label(const ModelInfo & model);
const ModelInfo *             find_model_by_quant(const std::string & quant);
const ModelInfo *             find_model_by_rank(int rank);
const ModelInfo *             find_model_by_filename(const std::string & filename);
