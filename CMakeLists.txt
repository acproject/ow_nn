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
    add_compile_options(-Wall -Wextra -O2)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /O2)
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 子目录
add_subdirectory(src)

# 主程序可执行
add_executable(ow_nn_main ${CMAKE_SOURCE_DIR}/main.cpp)
target_link_libraries(ow_nn_main PRIVATE ow_nn)
set_target_properties(ow_nn_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 导出编译命令（用于IDE）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)