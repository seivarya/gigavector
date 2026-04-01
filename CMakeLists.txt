cmake_minimum_required(VERSION 3.15)
project(GigaVector VERSION 0.8.2 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers (ASAN, TSAN, UBSAN)" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_NATIVE_OPTIMIZATIONS "Enable CPU-specific optimizations (-march=native -mtune=native)" OFF)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")

# CPU tuning is opt-in to avoid shipping binaries with unsupported instruction sets.
if(ENABLE_NATIVE_OPTIMIZATIONS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
endif()

# Position Independent Code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(include)

# Source files
file(GLOB GV_SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gv_*.c")
list(SORT GV_SOURCE_FILES)
set(SOURCES ${GV_SOURCE_FILES})

# Create library
add_library(GigaVector ${SOURCES})

# Check for libcurl (required for LLM support)
find_package(CURL)
if(CURL_FOUND)
    target_compile_definitions(GigaVector PRIVATE HAVE_CURL)
    target_link_libraries(GigaVector PRIVATE ${CURL_LIBRARIES})
    target_include_directories(GigaVector PRIVATE ${CURL_INCLUDE_DIRS})
    message(STATUS "  LLM support: Enabled (libcurl found)")
else()
    message(STATUS "  LLM support: Disabled (libcurl not found)")
    message(WARNING "Install libcurl-dev/libcurl-devel to enable LLM-based memory extraction")
endif()

# Check for libmicrohttpd (required for HTTP server)
find_library(MICROHTTPD_LIBRARY NAMES microhttpd)
find_path(MICROHTTPD_INCLUDE_DIR NAMES microhttpd.h)
if(MICROHTTPD_LIBRARY AND MICROHTTPD_INCLUDE_DIR)
    target_compile_definitions(GigaVector PRIVATE HAVE_MICROHTTPD)
    target_link_libraries(GigaVector PRIVATE ${MICROHTTPD_LIBRARY})
    target_include_directories(GigaVector PRIVATE ${MICROHTTPD_INCLUDE_DIR})
    message(STATUS "  HTTP Server: Enabled (libmicrohttpd found)")
else()
    message(STATUS "  HTTP Server: Disabled (libmicrohttpd not found)")
    message(WARNING "Install libmicrohttpd-dev to enable HTTP REST server")
endif()

# Check for CUDA (optional, for GPU acceleration)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(ENABLE_CUDA "Enable CUDA GPU acceleration" ON)
    if(ENABLE_CUDA)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        target_sources(GigaVector PRIVATE src/gv_gpu_kernels.cu)
        target_compile_definitions(GigaVector PRIVATE HAVE_CUDA)
        set_target_properties(GigaVector PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89;90"
        )
        message(STATUS "  GPU Acceleration: Enabled (CUDA found)")
    else()
        message(STATUS "  GPU Acceleration: Disabled (CUDA disabled)")
    endif()
else()
    message(STATUS "  GPU Acceleration: Disabled (CUDA not found)")
endif()

# Link libraries
target_link_libraries(GigaVector PRIVATE m pthread)

# Set library properties
set_target_properties(GigaVector PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/gigavector/gigavector.h"
)

# Install library
install(TARGETS GigaVector
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/gigavector
)

# Install headers
install(DIRECTORY include/gigavector/
    DESTINATION include/gigavector
    FILES_MATCHING PATTERN "*.h"
)

# Main executable
add_executable(gigavector_main main.c)
target_link_libraries(gigavector_main PRIVATE GigaVector)

# Install main executable
install(TARGETS gigavector_main
    RUNTIME DESTINATION bin
)

# CLI Tools
add_executable(gvbackup tools/gv_backup_cli.c)
target_link_libraries(gvbackup PRIVATE GigaVector)

add_executable(gvrestore tools/gv_restore_cli.c)
target_link_libraries(gvrestore PRIVATE GigaVector)

add_executable(gvinspect tools/gv_inspect_cli.c)
target_link_libraries(gvinspect PRIVATE GigaVector)

# Install CLI tools
install(TARGETS gvbackup gvrestore gvinspect
    RUNTIME DESTINATION bin
)

# Sanitizers
if(ENABLE_SANITIZERS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=thread -fsanitize=undefined -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=thread -fsanitize=undefined")
endif()

# Coverage
if(ENABLE_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -O0")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.c")
    list(SORT TEST_SOURCES)

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        string(REPLACE "test_" "" TEST_NAME ${TEST_NAME})
        add_executable(test_${TEST_NAME} ${TEST_SRC})
        target_link_libraries(test_${TEST_NAME} PRIVATE GigaVector m pthread)
        add_test(NAME ${TEST_NAME} COMMAND test_${TEST_NAME})
    endforeach()
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    set(BENCHMARK_SOURCES
        benchmarks/benchmark_simd.c
        benchmarks/benchmark_compare.c
        benchmarks/benchmark_ivfpq.c
        benchmarks/benchmark_ivfpq_recall.c
    )
    
    foreach(BENCH_SRC ${BENCHMARK_SOURCES})
        get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)
        add_executable(${BENCH_NAME} ${BENCH_SRC})
        target_link_libraries(${BENCH_NAME} PRIVATE GigaVector m)
    endforeach()
endif()

# Python bindings (optional, requires Python development headers)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Note: Python bindings are typically built via setup.py
    # This section can be extended to build Python extensions directly
    message(STATUS "Python bindings should be built using: cd python && pip install .")
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "GigaVector Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Native optimizations: ${ENABLE_NATIVE_OPTIMIZATIONS}")
message(STATUS "")
