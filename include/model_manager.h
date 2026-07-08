#pragma once

#include "./app_config.h"
#include "./hardware_profile.h"
#include "./model_catalog.h"

#include <string>
#include <vector>

struct ModelStatus {
    ModelInfo   info;
    bool        cached = false;
    std::string path;
    bool        recommended = false;
    bool        last_used = false;
    bool        fits = false;
    bool        downloadable = true;
    bool        catalog_model = true;
};

struct ModelDecision {
    HardwareProfile         hardware;
    std::vector<ModelStatus> models;
    int                     recommended_index = 0;
    std::string             selected_accelerator = "cpu";
    std::string             reason;
};

std::string   hf_cache_dir();
ModelDecision build_model_decision(const AppConfig & config);
bool          download_public_model(const ModelInfo & model, std::string & output_path, std::string & error);
bool          save_last_used_model(const ModelInfo & model, const std::string & path, std::string & error);
bool          describe_local_model_path(const std::string & path, ModelInfo & model);
std::string   cached_model_path(const std::string & repo, const std::string & filename);
