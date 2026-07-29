#include "../include/detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

constexpr int kMaxWindowContextTokens = 256;

struct DiscrepancySums {
    double log_likelihood = 0.0;
    double expected       = 0.0;
    double variance       = 0.0;
};

double discrepancy_from_sums(const DiscrepancySums & sums) {
    if (!std::isfinite(sums.log_likelihood) || !std::isfinite(sums.expected) || !std::isfinite(sums.variance) ||
        sums.variance <= 1e-9) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (sums.log_likelihood - sums.expected) / std::sqrt(sums.variance);
}

}  // namespace

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

namespace {

bool accumulate_discrepancy(const std::vector<float *> &     logits,
                            const std::vector<llama_token> & labels,
                            const int                        vocab_size,
                            DiscrepancySums &                sums) {
    if (vocab_size <= 0 || labels.empty() || logits.size() < labels.size()) {
        return false;
    }

    std::vector<double> buffer(vocab_size);
    for (std::size_t index = 0; index < labels.size(); ++index) {
        const auto stats = compute_token_stats(vocab_size, labels[index], logits[index], buffer);
        if (!std::isfinite(stats.log_likelihood) || !std::isfinite(stats.mean) || !std::isfinite(stats.variance)) {
            return false;
        }
        sums.log_likelihood += stats.log_likelihood;
        sums.expected += stats.mean;
        sums.variance += stats.variance;
    }
    return true;
}

}  // namespace

double compute_discrepancy(const std::vector<float *> &     all_logits,
                           const std::vector<llama_token> & tokens,
                           int                              vocab_size) {
    if (vocab_size <= 0 || tokens.size() < 2 || all_logits.size() < tokens.size() - 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::vector<llama_token> labels(tokens.begin() + 1, tokens.end());
    DiscrepancySums               sums;
    if (!accumulate_discrepancy(all_logits, labels, vocab_size, sums)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Positive discrepancy means the observed tokens are more probable than
    // alternatives sampled from the model: the model-like direction defined
    // by Fast-DetectGPT.
    return discrepancy_from_sums(sums);
}

AnalysisResult analyze_text_detailed(const LlamaState & llama, const std::string & text, const int n_ctx) {
    AnalysisResult result;

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

    const int context_capacity =
        std::min({ n_ctx, static_cast<int>(llama_n_ctx(llama.ctx)), static_cast<int>(llama_n_batch(llama.ctx)) });
    if (context_capacity < 2 || (n_tokens > context_capacity && context_capacity < 3)) {
        result.error = "The model context and batch size are too small for windowed scoring.";
        return result;
    }

    const int overlap =
        std::min({ kMaxWindowContextTokens, std::max(1, (context_capacity - 1) / 4), context_capacity - 2 });
    const int         vocab_size = llama_vocab_n_tokens(llama.vocab);
    const llama_token bos_token  = llama_vocab_bos(llama.vocab);
    const bool        has_bos     = bos_token != LLAMA_TOKEN_NULL && tokens.front() == bos_token;

    const auto started_at = std::chrono::steady_clock::now();
    DiscrepancySums sums;
    int             target_start = 1;
    while (target_start < n_tokens) {
        const bool first_window = target_start == 1;
        const int  prefix_tokens = !first_window && has_bos ? 1 : 0;
        const int  content_floor = has_bos ? 1 : 0;
        const int  context_start = first_window ? 0 : std::max(content_floor, target_start - overlap);
        const int  content_capacity = context_capacity - prefix_tokens;
        const int  window_end        = std::min(n_tokens, context_start + content_capacity);
        const int  scored_tokens     = window_end - target_start;
        if (scored_tokens <= 0) {
            result.error = "Could not advance the windowed scorer.";
            return result;
        }

        std::vector<llama_token> window_tokens;
        window_tokens.reserve(prefix_tokens + window_end - context_start);
        if (prefix_tokens == 1) {
            window_tokens.push_back(bos_token);
        }
        window_tokens.insert(window_tokens.end(), tokens.begin() + context_start, tokens.begin() + window_end);

        const int local_target_start = first_window ? 1 : prefix_tokens + target_start - context_start;
        const int first_predictor    = local_target_start - 1;

        // Each window starts from a clean KV cache. Later windows prepend BOS
        // and prior tokens as context, but request logits only for new targets.
        const auto memory = llama_get_memory(llama.ctx);
        llama_memory_seq_rm(memory, -1, -1, -1);

        auto batch     = llama_batch_init(static_cast<int>(window_tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int>(window_tokens.size());
        for (int index = 0; index < batch.n_tokens; ++index) {
            batch.token[index]     = window_tokens[index];
            batch.pos[index]       = index;
            batch.n_seq_id[index]  = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index]    = index >= first_predictor && index < first_predictor + scored_tokens;
        }

        if (llama_decode(llama.ctx, batch) != 0) {
            result.error = "Inference failed while scoring context window " + std::to_string(result.windows + 1) + ".";
            llama_batch_free(batch);
            return result;
        }

        std::vector<float *> logits;
        logits.reserve(scored_tokens);
        for (int index = 0; index < scored_tokens; ++index) {
            logits.push_back(llama_get_logits_ith(llama.ctx, first_predictor + index));
        }
        const std::vector<llama_token> labels(tokens.begin() + target_start, tokens.begin() + window_end);
        if (!accumulate_discrepancy(logits, labels, vocab_size, sums)) {
            result.error = "Could not compute discrepancy statistics for context window " +
                           std::to_string(result.windows + 1) + ".";
            llama_batch_free(batch);
            return result;
        }

        llama_batch_free(batch);
        ++result.windows;
        target_start = window_end;
    }

    const double score    = discrepancy_from_sums(sums);
    const auto   ended_at = std::chrono::steady_clock::now();
    if (!std::isfinite(score)) {
        result.error = "Could not compute a finite discrepancy score.";
        return result;
    }

    result.ok                = true;
    result.discrepancy       = score;
    result.context_overlap   = result.windows > 1 ? overlap : 0;
    result.elapsed_seconds   = std::chrono::duration<double>(ended_at - started_at).count();
    result.tokens_per_second =
        result.elapsed_seconds > 0.0 ? static_cast<double>(result.tokens) / result.elapsed_seconds : 0.0;
    if (result.tokens < kRecommendedMinScoredTokens) {
        result.warning = "Low confidence: short text (" + std::to_string(result.tokens) + " scored tokens; at least " +
                         std::to_string(kRecommendedMinScoredTokens) + " recommended).";
    }
    if (result.windows > 1) {
        if (!result.warning.empty()) {
            result.warning += " ";
        }
        result.warning += "Windowed analysis: " + std::to_string(result.windows) + " windows with up to " +
                          std::to_string(result.context_overlap) + " preceding context tokens.";
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
