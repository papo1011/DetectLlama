cmake_minimum_required(VERSION 3.15)
project(DetectLlama)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(DETECT_LLAMA_GPU "auto" CACHE STRING "GPU backend for bundled llama.cpp: auto, cpu, cuda, metal, vulkan")
set_property(CACHE DETECT_LLAMA_GPU PROPERTY STRINGS auto cpu cuda metal vulkan)

# ------ argparse ------
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)

FetchContent_MakeAvailable(argparse)

# ------ llama.cpp ------
find_package(llama QUIET)

if (llama_FOUND)
        message(STATUS "Using system llama.cpp package")
else()
        message(STATUS "System llama.cpp package not found; fetching llama.cpp")

        set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
        set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
        set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

        if (DETECT_LLAMA_GPU STREQUAL "cpu")
                set(GGML_CUDA OFF CACHE BOOL "" FORCE)
                set(GGML_METAL OFF CACHE BOOL "" FORCE)
                set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
                message(STATUS "Bundled llama.cpp GPU backend: CPU only")
        elseif (DETECT_LLAMA_GPU STREQUAL "cuda")
                set(GGML_CUDA ON CACHE BOOL "" FORCE)
                message(STATUS "Bundled llama.cpp GPU backend: CUDA")
        elseif (DETECT_LLAMA_GPU STREQUAL "metal")
                set(GGML_METAL ON CACHE BOOL "" FORCE)
                message(STATUS "Bundled llama.cpp GPU backend: Metal")
        elseif (DETECT_LLAMA_GPU STREQUAL "vulkan")
                set(GGML_VULKAN ON CACHE BOOL "" FORCE)
                message(STATUS "Bundled llama.cpp GPU backend: Vulkan")
        elseif (DETECT_LLAMA_GPU STREQUAL "auto")
                if (APPLE)
                        set(GGML_METAL ON CACHE BOOL "" FORCE)
                        message(STATUS "Bundled llama.cpp GPU backend: Metal auto-detected")
                else()
                        find_program(DETECT_LLAMA_NVCC nvcc)
                        if (DETECT_LLAMA_NVCC)
                                set(GGML_CUDA ON CACHE BOOL "" FORCE)
                                message(STATUS "Bundled llama.cpp GPU backend: CUDA auto-detected")
                        else()
                                message(STATUS "Bundled llama.cpp GPU backend: CPU fallback")
                        endif()
                endif()
        else()
                message(FATAL_ERROR "Unsupported DETECT_LLAMA_GPU='${DETECT_LLAMA_GPU}'. Use auto, cpu, cuda, metal, or vulkan.")
        endif()

        FetchContent_Declare(
                llama_cpp
                GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
                GIT_TAG master
                GIT_SHALLOW TRUE
                GIT_PROGRESS TRUE
        )

        FetchContent_MakeAvailable(llama_cpp)
endif()

add_executable(DetectLlama src/main.cpp
        src/detect.cpp
        src/utils.cpp
        src/io.cpp
        include/detect.h
        include/utils.h
        include/io.h
)

target_link_libraries(DetectLlama PRIVATE llama argparse::argparse)
