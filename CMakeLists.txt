cmake_minimum_required(VERSION 3.16)
project(ow_nn VERSION 0.1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 获取 nlohmann_json
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2 # choose a stable tag
)
FetchContent_MakeAvailable(json)

# 编译选项
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -O2 -mavx2 -mfma)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /arch:AVX2)
    # 在Release模式下启用优化
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 子目录
add_subdirectory(src)

# Add benchmark executable
add_executable(benchmark_matmul benchmark_matmul.cpp)
target_compile_features(benchmark_matmul PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(benchmark_matmul PRIVATE /arch:AVX2 /openmp)
    set_target_properties(benchmark_matmul PROPERTIES
        COMPILE_FLAGS_RELEASE "/O2"
        COMPILE_FLAGS_DEBUG "/Od"
    )
else()
    target_compile_options(benchmark_matmul PRIVATE -O3 -march=native -fopenmp)
    target_link_libraries(benchmark_matmul PRIVATE -fopenmp)
endif()
target_include_directories(benchmark_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 主程序可执行
add_executable(ow_nn_main ${CMAKE_SOURCE_DIR}/main.cpp)
target_link_libraries(ow_nn_main PRIVATE ow_nn)
set_target_properties(ow_nn_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# 在链接前确保输出目录存在（VS 多配置生成器）
add_custom_command(TARGET ow_nn_main PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ow_nn_main>"
)

# 导出编译命令（用于IDE）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)