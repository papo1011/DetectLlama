#include "../include/signals.h"

#include <csignal>
#include <cstdlib>

std::atomic<bool> g_interrupted(false);

void signal_handler(const int signum) {
    g_interrupted = true;
}

void fatal_signal_handler(const int signum) {
    std::_Exit(128 + signum);
}

void install_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGABRT, fatal_signal_handler);
}
