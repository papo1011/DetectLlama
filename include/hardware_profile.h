#pragma once

#include <string>

struct HardwareProfile {
    std::string os_name;
    std::string arch_name;
    std::string accelerator = "cpu";
    std::string gpu_name;
    int         cpu_cores = 1;
    long long   total_ram_mb = 0;
    long long   available_ram_mb = 0;
    long long   gpu_total_mb = 0;
    long long   gpu_free_mb = 0;
    long long   disk_free_mb = 0;
    long long   memory_pool_mb = 0;
};

HardwareProfile detect_hardware_profile(const std::string & cache_dir);
