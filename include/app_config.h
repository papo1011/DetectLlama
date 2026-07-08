#pragma once

struct AppConfig {
    int         n_ctx = 2048;
    int         n_batch = 2048;
    int         target_tokens_per_sec = 30;
    bool        use_gpu = false;
};
