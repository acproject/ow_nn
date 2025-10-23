# 选项：图像/视频后端（默认轻量，macOS 上可用 Apple ImageIO）
if(APPLE)
    option(OW_WITH_APPLE_IMAGEIO "Use Apple ImageIO for image loading" ON)
else()
    option(OW_WITH_APPLE_IMAGEIO "Use Apple ImageIO for image loading" OFF)
endif()
option(OW_WITH_STB "Use stb_image.h for image loading" OFF)
option(OW_WITH_WIC "Use Windows Imaging Component for image loading" OFF)
option(OW_WITH_OPENCV "Enable OpenCV for image ops" OFF)

# 核心库源文件
set(SOURCES
    tokenizer.cpp
    transformer.cpp
    backend_qwen3vl.hpp
    sampler.cpp
    context.hpp
    thread_pool.hpp
    tensor.hpp
    cgraph.hpp
    safetensors_loader.hpp
    lazy_safetensors_loader.hpp
    memory_optimizer.hpp
    ops.hpp
    logging.hpp
    utils.hpp
    image_io.hpp
)

# 创建静态库
add_library(ow_nn STATIC ${SOURCES})

# 链接数学库（在Unix系统上）
if(UNIX)
    target_link_libraries(ow_nn PRIVATE m)
endif()

# macOS: 链接 Apple ImageIO 框架
if(APPLE AND OW_WITH_APPLE_IMAGEIO)
    target_compile_definitions(ow_nn PRIVATE OW_WITH_APPLE_IMAGEIO)
    target_link_libraries(ow_nn PRIVATE "-framework CoreGraphics" "-framework ImageIO" "-framework CoreFoundation")
endif()

# Windows: 启用 WIC（Windows Imaging Component）
if(WIN32 AND OW_WITH_WIC)
    target_compile_definitions(ow_nn PRIVATE OW_WITH_WIC)
    target_link_libraries(ow_nn PRIVATE Ole32 Windowscodecs)
endif()

# stb 支持（用户需在包含路径中提供 <stb_image.h>）
if(OW_WITH_STB)
    target_compile_definitions(ow_nn PRIVATE OW_WITH_STB)
    # 提供一个 TU 编译 stb 实现（需要头文件可见）
    target_sources(ow_nn PRIVATE stb_image_impl.cpp)
endif()

# 链接 nlohmann_json
target_link_libraries(ow_nn PRIVATE nlohmann_json::nlohmann_json)

# 设置目标属性
set_target_properties(ow_nn PROPERTIES
    OUTPUT_NAME "ow_nn"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)