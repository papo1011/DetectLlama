#pragma once

#include <string>
#include <vector>

struct ModelInfo {
    std::string quant;
    std::string filename;
    int         size_mb = 0;
    int         rank = 0;
};

const std::vector<ModelInfo> & model_catalog();
const ModelInfo *             find_model_by_quant(const std::string & quant);
const ModelInfo *             find_model_by_rank(int rank);
