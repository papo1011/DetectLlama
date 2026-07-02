#pragma once

#include <atomic>

extern std::atomic<bool> g_interrupted;

void signal_handler(int signum);
