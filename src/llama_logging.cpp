#include "../include/llama_logging.h"

#include <cstdio>

void custom_log(ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}
