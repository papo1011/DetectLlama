#pragma once

#include "./hardware_profile.h"

#include <string>

struct ModelInfo {
    std::string family;
    std::string quant;
    std::string repo;
    std::string filename;
    int         size_mb = 0;
    double      threshold = 0.0;
    int         threshold_context = 0;
    std::string calibration_id;
};

struct ModelState {
    HardwareProfile hardware;
    std::string     path;
    bool            cached = false;
    bool            fits = false;
};

const ModelInfo & detectllama_model();
std::string       model_label();
std::string       hf_cache_dir();
ModelState        inspect_model_state();
bool              download_model(std::string & output_path, std::string & error);
bool              is_detectllama_model_path(const std::string & path);
std::string       cached_model_path();
