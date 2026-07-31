#pragma once

inline constexpr int kContextLength = 128;
inline constexpr int kBatchSize     = 128;
inline constexpr double kDetectionThreshold = -1.550;
inline constexpr const char * kCalibrationId =
    "ghostbuster-953b44b7-seed20260731-ctx128-q4_0-claude-gpt";

struct AppConfig {
    bool use_gpu = false;
};
