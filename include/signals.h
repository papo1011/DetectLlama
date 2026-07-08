#pragma once

#include <atomic>

extern std::atomic<bool> g_interrupted;

void install_signal_handlers();
void signal_handler(int signum);
void fatal_signal_handler(int signum);
