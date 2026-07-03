#include "../include/detect.h"

#include <chrono>
#include <cmath>
#include <iostream>

TokenStats compute_token_stats(const int             vocab_size,
                               const int             token_id,
                               const float *         logits,
                               std::vector<double> & buffer) {
    float max_logit = -1e9;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        buffer[i] = std::exp(logits[i] - max_logit);
        sum_exp += buffer[i];
    }
    double log_sum_exp = std::log(sum_exp);

    TokenStats stats = { 0.0, 0.0, 0.0 };

    if (token_id >= 0 && token_id < vocab_size) {
        stats.log_likelihood = (logits[token_id] - max_logit) - log_sum_exp;
    }

    double mean            = 0.0;  // E[X]
    double expected_square = 0.0;  // E[X^2]

    for (int i = 0; i < vocab_size; i++) {
        double p  = buffer[i] / sum_exp;
        double lp = (logits[i] - max_logit) - log_sum_exp;

        mean += p * lp;
        expected_square += p * (lp * lp);
    }

    stats.mean     = mean;
    stats.variance = expected_square - (mean * mean);  // E[X^2] - (E[X])^2

    return stats;
}

double compute_discrepancy(const std::vector<float *> &     all_logits,
                           const std::vector<llama_token> & tokens,
                           int                              vocab_size) {
    double sum_ll   = 0.0;
    double sum_mean = 0.0;
    double sum_var  = 0.0;

    std::vector<double> buffer(vocab_size);

    const size_t steps = tokens.size() - 1;

    for (size_t t = 0; t < steps; t++) {
        const int     token_id = tokens[t + 1];
        const float * logits   = all_logits[t];

        auto [log_likelihood, mean, variance] = compute_token_stats(vocab_size, token_id, logits, buffer);

        sum_ll += log_likelihood;
        sum_mean += mean;
        sum_var += variance;
    }

    if (sum_var <= 1e-9) {
        return 0.0;
    }

    return (sum_ll - sum_mean) / std::sqrt(sum_var);
}

AnalysisResult analyze_text_detailed(const LlamaState & llama, const std::string & text, const int n_ctx) {
    AnalysisResult result;

    // clear cache
    const auto memory = llama_get_memory(llama.ctx);
    llama_memory_seq_rm(memory, -1, -1, -1);

    // Input text tokenized
    // size: input text length + 2 for BOS and EOS
    std::vector<llama_token> tokens(text.length() + 2);
    int n_tokens = llama_tokenize(llama.vocab, text.c_str(), static_cast<int>(text.length()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, false);

    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(llama.vocab, text.c_str(), static_cast<int>(text.length()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, false);
    }
    tokens.resize(n_tokens);
    result.tokens = n_tokens;

    if (n_tokens < 2) {
        result.error = "Not enough tokens provided (minimum 2 tokens)";
        return result;
    }

    if (n_tokens > n_ctx) {
        result.error = "Too many tokens provided: " + std::to_string(n_tokens) + " (maximum " + std::to_string(n_ctx) + ")";
        return result;
    }

    auto batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.n_tokens     = n_tokens;
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }

    const auto started_at = std::chrono::steady_clock::now();
    if (llama_decode(llama.ctx, batch) != 0) {
        result.error = "Inference failed";
        llama_batch_free(batch);
        return result;
    }

    std::vector<float *> logits_ptrs;
    logits_ptrs.reserve(n_tokens);  // reserve space avoiding reallocations
    for (int i = 0; i < n_tokens; i++) {
        logits_ptrs.push_back(llama_get_logits_ith(llama.ctx, i));
    }

    const int    vocab_size = llama_vocab_n_tokens(llama.vocab);
    const double score      = compute_discrepancy(logits_ptrs, tokens, vocab_size);
    const auto   ended_at   = std::chrono::steady_clock::now();

    llama_batch_free(batch);

    result.ok                = true;
    result.discrepancy       = score;
    result.elapsed_seconds   = std::chrono::duration<double>(ended_at - started_at).count();
    result.tokens_per_second = result.elapsed_seconds > 0.0 ? static_cast<double>(n_tokens) / result.elapsed_seconds : 0.0;
    return result;
}

double analyze_text(const LlamaState & llama, const std::string & text, const int n_ctx) {
    const auto result = analyze_text_detailed(llama, text, n_ctx);
    if (!result.ok) {
        std::cerr << result.error << std::endl;
        return 0.0;
    }
    return result.discrepancy;
}
