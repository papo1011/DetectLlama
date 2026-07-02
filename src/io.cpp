#include "../include/io.h"

#include <fstream>
#include <sstream>

bool read_file_to_string(const std::string & path, std::string & out) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}
