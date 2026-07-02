cmake_minimum_required(VERSION 3.15)
project(fast-detect-gpt)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# ------ argparse ------
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)

FetchContent_MakeAvailable(argparse)

# ------ llama.cpp ------
find_package(llama REQUIRED)

# ------ arrow ------
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

add_executable(fast-detect-gpt src/main.cpp
        src/detect.cpp
        src/utils.cpp
        src/io.cpp
        include/detect.h
        include/utils.h
        include/io.h
)

target_link_libraries(fast-detect-gpt PRIVATE llama argparse::argparse Arrow::arrow_shared Parquet::parquet_shared)
