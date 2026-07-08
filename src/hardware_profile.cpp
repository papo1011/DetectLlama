#include "../include/hardware_profile.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace {

std::string trim_copy(const std::string & value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string command_output(const std::string & command) {
    std::array<char, 256> buffer{};
    std::string          output;
    FILE *               pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "";
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    pclose(pipe);
    return trim_copy(output);
}

std::string uname_value(const std::string & flag) {
    return command_output("uname " + flag + " 2>/dev/null");
}

long long linux_mem_mb(const std::string & key) {
    std::ifstream meminfo("/proc/meminfo");
    std::string   name;
    long long     kb = 0;
    std::string   unit;

    while (meminfo >> name >> kb >> unit) {
        if (name == key + ":") {
            return kb / 1024;
        }
    }
    return 0;
}

long long mac_total_mem_mb() {
    const std::string bytes = command_output("sysctl -n hw.memsize 2>/dev/null");
    if (!bytes.empty()) {
        return std::stoll(bytes) / 1024 / 1024;
    }

    const std::string hostinfo_mb = command_output(
        "hostinfo 2>/dev/null | awk '/Primary memory available:/ { value=$4; unit=$5; if (unit ~ /gigabytes/) printf \"%d\", value * 1024; else if (unit ~ /megabytes/) printf \"%d\", value; }'");
    if (!hostinfo_mb.empty()) {
        return std::stoll(hostinfo_mb);
    }

    const std::string profiler_mb = command_output(
        "system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Memory:/ { split($2, parts, \" \"); if (parts[2] == \"GB\") printf \"%d\", parts[1] * 1024; else if (parts[2] == \"MB\") printf \"%d\", parts[1]; exit }'");
    if (!profiler_mb.empty()) {
        return std::stoll(profiler_mb);
    }

    return 0;
}

long long mac_available_mem_mb() {
    const std::string output = command_output("pagesize 2>/dev/null");
    const long long   page_size = output.empty() ? 4096 : std::stoll(output);
    const std::string vm = command_output("vm_stat 2>/dev/null");
    std::istringstream input(vm);
    std::string        line;
    long long          free_pages = 0;
    long long          inactive_pages = 0;
    long long          speculative_pages = 0;

    while (std::getline(input, line)) {
        auto read_pages = [](std::string value) {
            value.erase(std::remove(value.begin(), value.end(), '.'), value.end());
            value = trim_copy(value);
            return value.empty() ? 0LL : std::stoll(value);
        };

        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }

        const std::string label = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (label == "Pages free") {
            free_pages = read_pages(value);
        } else if (label == "Pages inactive") {
            inactive_pages = read_pages(value);
        } else if (label == "Pages speculative") {
            speculative_pages = read_pages(value);
        }
    }

    return (free_pages + inactive_pages + speculative_pages) * page_size / 1024 / 1024;
}

bool detect_nvidia(HardwareProfile & profile) {
    const std::string output = command_output(
        "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null");
    if (output.empty()) {
        return false;
    }

    std::istringstream lines(output);
    std::string        line;
    long long          best_free = 0;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        std::string        name;
        std::string        total_text;
        std::string        free_text;
        if (!std::getline(fields, name, ',') || !std::getline(fields, total_text, ',') || !std::getline(fields, free_text, ',')) {
            continue;
        }

        const long long total = std::stoll(trim_copy(total_text));
        const long long free = std::stoll(trim_copy(free_text));
        if (free > best_free) {
            best_free = free;
            profile.gpu_name = trim_copy(name);
            profile.gpu_total_mb = total;
            profile.gpu_free_mb = free;
        }
    }

    if (best_free <= 0) {
        return false;
    }

    profile.accelerator = "nvidia";
    return true;
}

std::string detect_apple_gpu_name() {
    std::string value = command_output("system_profiler SPHardwareDataType SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chip:/ && chip == \"\" { chip=$2 } /Chipset Model:/ && gpu == \"\" { gpu=$2 } END { if (gpu != \"\") print gpu; else if (chip != \"\") print chip }'");
    return value.empty() ? "Apple Silicon" : value;
}

}  // namespace

HardwareProfile detect_hardware_profile(const std::string & cache_dir) {
    HardwareProfile profile;
    profile.os_name = uname_value("-s");
    profile.arch_name = uname_value("-m");
    profile.cpu_cores = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    std::error_code fs_error;
    std::filesystem::create_directories(cache_dir, fs_error);
    const auto space = std::filesystem::space(cache_dir, fs_error);
    profile.disk_free_mb = fs_error ? 0 : static_cast<long long>(space.available / 1024 / 1024);

    if (profile.os_name == "Darwin") {
        profile.total_ram_mb = mac_total_mem_mb();
        profile.available_ram_mb = mac_available_mem_mb();
        if (profile.arch_name == "arm64") {
            profile.accelerator = "apple-unified";
            profile.gpu_name = detect_apple_gpu_name();
            profile.gpu_total_mb = profile.total_ram_mb;
            profile.gpu_free_mb = profile.available_ram_mb;
        }
    } else {
        profile.total_ram_mb = linux_mem_mb("MemTotal");
        profile.available_ram_mb = linux_mem_mb("MemAvailable");
        detect_nvidia(profile);
    }

    if (profile.total_ram_mb == 0) {
        profile.total_ram_mb = profile.available_ram_mb;
    }

    long long ram_soft_pool_mb = profile.total_ram_mb * 75 / 100;
    if (profile.available_ram_mb > ram_soft_pool_mb) {
        ram_soft_pool_mb = profile.available_ram_mb;
    }

    profile.memory_pool_mb = ram_soft_pool_mb;
    if (profile.accelerator == "apple-unified") {
        profile.memory_pool_mb = profile.total_ram_mb * 90 / 100;
    } else if (profile.accelerator == "nvidia") {
        profile.memory_pool_mb = profile.gpu_free_mb;
    }

    return profile;
}
