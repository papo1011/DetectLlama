cmake_minimum_required(VERSION 3.15)
project(DetectLlama)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(DETECT_LLAMA_GPU "auto" CACHE STRING "GPU backend for bundled llama.cpp: auto, cpu, cuda, metal, vulkan")
set_property(CACHE DETECT_LLAMA_GPU PROPERTY STRINGS auto cpu cuda metal vulkan)
set(DETECT_LLAMA_CPP_DIR "" CACHE PATH "Path to a local llama.cpp source checkout")
option(DETECT_LLAMA_FETCH "Download llama.cpp when no local installation is available" OFF)

# ------ llama.cpp ------
function(detect_llama_configure_bundled_backend)
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
        set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
        set(LLAMA_OPENSSL ON CACHE BOOL "" FORCE)

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
endfunction()

if (DETECT_LLAMA_CPP_DIR)
        get_filename_component(
                DETECT_LLAMA_CPP_DIR_ABSOLUTE
                "${DETECT_LLAMA_CPP_DIR}"
                ABSOLUTE
                BASE_DIR "${CMAKE_SOURCE_DIR}"
        )

        if (NOT EXISTS "${DETECT_LLAMA_CPP_DIR_ABSOLUTE}/CMakeLists.txt" OR
            NOT EXISTS "${DETECT_LLAMA_CPP_DIR_ABSOLUTE}/include/llama.h")
                message(FATAL_ERROR
                        "DETECT_LLAMA_CPP_DIR is not a llama.cpp source checkout: "
                        "${DETECT_LLAMA_CPP_DIR_ABSOLUTE}"
                )
        endif()

        set(DETECT_LLAMA_CPP_DIR "${DETECT_LLAMA_CPP_DIR_ABSOLUTE}" CACHE PATH
                "Path to a local llama.cpp source checkout" FORCE)
        message(STATUS "Using local llama.cpp source: ${DETECT_LLAMA_CPP_DIR}")
        detect_llama_configure_bundled_backend()
        add_subdirectory(
                "${DETECT_LLAMA_CPP_DIR}"
                "${CMAKE_BINARY_DIR}/_deps/llama_cpp-local-build"
        )
        set(DETECT_LLAMA_RESOLVED_PROVIDER "local" CACHE INTERNAL
                "Resolved llama.cpp provider" FORCE)
else()
        find_package(llama QUIET)

        if (llama_FOUND)
                message(STATUS "Using system llama.cpp package")
                set(DETECT_LLAMA_RESOLVED_PROVIDER "system" CACHE INTERNAL
                        "Resolved llama.cpp provider" FORCE)
        elseif (DETECT_LLAMA_FETCH)
                message(STATUS "System llama.cpp package not found; downloading llama.cpp")
                detect_llama_configure_bundled_backend()

                FetchContent_Declare(
                        llama_cpp
                        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
                        GIT_TAG master
                        GIT_SHALLOW TRUE
                        GIT_PROGRESS TRUE
                )

                FetchContent_MakeAvailable(llama_cpp)
                set(DETECT_LLAMA_RESOLVED_PROVIDER "download" CACHE INTERNAL
                        "Resolved llama.cpp provider" FORCE)
        else()
                message(FATAL_ERROR
                        "llama.cpp was not found.\n"
                        "Use ./build.sh to choose a local source checkout or approve the download.\n"
                        "For a non-interactive build, set DETECT_LLAMA_CPP_DIR or DETECT_LLAMA_FETCH=ON."
                )
        endif()
endif()

# ------ argparse ------
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)

FetchContent_MakeAvailable(argparse)

# ------ FTXUI ------
FetchContent_Declare(
        ftxui
        GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
        GIT_TAG v5.0.0
)

FetchContent_MakeAvailable(ftxui)

add_library(detectllama_core
        src/backend.cpp
        src/console.cpp
        src/detect.cpp
        src/hardware_profile.cpp
        src/io.cpp
        src/llama_logging.cpp
        src/llama_state.cpp
        src/model_catalog.cpp
        src/model_manager.cpp
        src/signals.cpp
        include/app_config.h
        include/backend.h
        include/console.h
        include/detect.h
        include/hardware_profile.h
        include/io.h
        include/llama_logging.h
        include/llama_state.h
        include/model_catalog.h
        include/model_manager.h
)

target_link_libraries(detectllama_core PUBLIC llama)
target_include_directories(detectllama_core PUBLIC include)

add_executable(DetectLlama
        src/main.cpp
        src/tui.cpp
        include/signals.h
        include/tui.h
)

target_link_libraries(DetectLlama PRIVATE detectllama_core argparse::argparse ftxui::screen ftxui::dom ftxui::component)

add_executable(DetectLlamaBackend
        src/backend_cli.cpp
)

target_link_libraries(DetectLlamaBackend PRIVATE detectllama_core argparse::argparse)
