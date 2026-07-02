#include "../include/signals.h"

#include <iostream>

std::atomic<bool> g_interrupted(false);

void signal_handler(const int signum) {
    std::cout << "\n\nInterrupt signal (" << signum << ") received." << std::endl;
    std::cout << "Interrupt requested." << std::endl;
    g_interrupted = true;
}
