#pragma once
#include "./llama_state.h"
#include "llama.h"

#include <string>
#include <vector>

struct TokenStats {
    double log_likelihood;
    double mean;
    double variance;
};

struct AnalysisResult {
    bool        ok                = false;
    double      discrepancy       = 0.0;
    int         tokens            = 0;
    double      elapsed_seconds   = 0.0;
    double      tokens_per_second = 0.0;
    std::string error;
};

TokenStats compute_token_stats(int vocab_size, int token_id, const float * logits, std::vector<double> & buffer);

double compute_discrepancy(const std::vector<float *> &     all_logits,
                           const std::vector<llama_token> & tokens,
                           int                              vocab_size);

AnalysisResult analyze_text_detailed(const LlamaState & llama, const std::string & text, int n_ctx);

double analyze_text(const LlamaState & llama, const std::string & text, int n_ctx);
