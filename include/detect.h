#pragma once
#include "./llama_state.h"
#include "llama.h"

#include <vector>

struct TokenStats {
    double log_likelihood;
    double mean;
    double variance;
};

TokenStats compute_token_stats(int vocab_size, int token_id, const float * logits, std::vector<double> & buffer);

double compute_discrepancy(const std::vector<float *> &     all_logits,
                           const std::vector<llama_token> & tokens,
                           int                              vocab_size);

double analyze_text(const LlamaState & llama, const std::string & text, int n_ctx);
