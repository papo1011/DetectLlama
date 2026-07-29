#include "../include/detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

TokenStats compute_token_stats(const int             vocab_size,
                               const int             token_id,
                               const float *         logits,
                               std::vector<double> & buffer) {
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if (vocab_size <= 0 || token_id < 0 || token_id >= vocab_size || logits == nullptr ||
        buffer.size() < static_cast<std::size_t>(vocab_size)) {
        return { invalid, invalid, invalid };
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; i++) {
        if (!std::isfinite(logits[i])) {
            return { invalid, invalid, invalid };
        }
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        buffer[i] = std::exp(logits[i] - max_logit);
        sum_exp += buffer[i];
    }
    if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
        return { invalid, invalid, invalid };
    }
    const double log_sum_exp = std::log(sum_exp);

    TokenStats stats = { (logits[token_id] - max_logit) - log_sum_exp, 0.0, 0.0 };

    double expected_square = 0.0;

    // The analytic Fast-DetectGPT statistic uses the model distribution to
    // compute E[log p(X)] and Var[log p(X)] without drawing token samples.
    for (int i = 0; i < vocab_size; i++) {
        const double probability     = buffer[i] / sum_exp;
        const double log_probability = (logits[i] - max_logit) - log_sum_exp;

        stats.mean += probability * log_probability;
        expected_square += probability * log_probability * log_probability;
    }

    // Round-off can make a theoretically non-negative variance slightly
    // negative, especially with quantized logits.
    stats.variance = std::max(0.0, expected_square - stats.mean * stats.mean);

    return stats;
}

double compute_discrepancy(const std::vector<float *> &     all_logits,
                           const std::vector<llama_token> & tokens,
                           int                              vocab_size) {
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if (vocab_size <= 0 || tokens.size() < 2 || all_logits.size() < tokens.size() - 1) {
        return invalid;
    }

    double sum_ll   = 0.0;
    double sum_mean = 0.0;
    double sum_var  = 0.0;

    std::vector<double> buffer(vocab_size);

    const size_t steps = tokens.size() - 1;

    for (size_t t = 0; t < steps; t++) {
        const int     token_id = tokens[t + 1];
        const float * logits   = all_logits[t];

        const auto [log_likelihood, mean, variance] = compute_token_stats(vocab_size, token_id, logits, buffer);
        if (!std::isfinite(log_likelihood) || !std::isfinite(mean) || !std::isfinite(variance)) {
            return invalid;
        }

        sum_ll += log_likelihood;
        sum_mean += mean;
        sum_var += variance;
    }

    if (!std::isfinite(sum_var) || sum_var <= 1e-9) {
        return invalid;
    }

    // Positive discrepancy means the observed tokens are more probable than
    // alternatives sampled from the model: the model-like direction defined
    // by Fast-DetectGPT.
    return (sum_ll - sum_mean) / std::sqrt(sum_var);
}

AnalysisResult analyze_text_detailed(const LlamaState & llama, const std::string & text, const int n_ctx) {
    AnalysisResult result;

    // Clear the KV cache so each passage is scored independently.
    const auto memory = llama_get_memory(llama.ctx);
    llama_memory_seq_rm(memory, -1, -1, -1);

    // A byte-sized buffer plus special-token headroom is a safe first attempt;
    // llama_tokenize reports the exact required size if it is insufficient.
    std::vector<llama_token> tokens(text.length() + 2);
    int n_tokens = llama_tokenize(llama.vocab, text.c_str(), static_cast<int>(text.length()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, false);

    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(llama.vocab, text.c_str(), static_cast<int>(text.length()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, false);
    }
    if (n_tokens < 0) {
        result.error = "Tokenization failed.";
        return result;
    }
    tokens.resize(n_tokens);
    result.tokens = std::max(0, n_tokens - 1);

    if (n_tokens < 2) {
        result.error = "Not enough text to score (at least one text token is required).";
        return result;
    }

    if (n_tokens > n_ctx) {
        result.error =
            "Too many tokens provided: " + std::to_string(n_tokens) + " (maximum " + std::to_string(n_ctx) + ")";
        return result;
    }

    auto batch     = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
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

    if (!std::isfinite(score)) {
        result.error = "Could not compute a finite discrepancy score.";
        return result;
    }

    result.ok              = true;
    result.discrepancy     = score;
    result.elapsed_seconds = std::chrono::duration<double>(ended_at - started_at).count();
    result.tokens_per_second =
        result.elapsed_seconds > 0.0 ? static_cast<double>(result.tokens) / result.elapsed_seconds : 0.0;
    if (result.tokens < kRecommendedMinScoredTokens) {
        result.warning = "Low confidence: short text (" + std::to_string(result.tokens) + " scored tokens; at least " +
                         std::to_string(kRecommendedMinScoredTokens) + " recommended).";
    }
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
