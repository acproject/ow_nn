# 核心库源文件
set(SOURCES
    tokenizer.cpp
    transformer.cpp
    context.hpp
    thread_pool.hpp
    tensor.hpp
    cgraph.hpp
    safetensors_loader.hpp
    ops.hpp
)

# 创建静态库
add_library(ow_nn STATIC ${SOURCES})

# 链接数学库（在Unix系统上）
if(UNIX)
    target_link_libraries(ow_nn PRIVATE m)
endif()

# 链接 nlohmann_json
target_link_libraries(ow_nn PRIVATE nlohmann_json::nlohmann_json)

# 设置目标属性
set_target_properties(ow_nn PROPERTIES
    OUTPUT_NAME "ow_nn"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)