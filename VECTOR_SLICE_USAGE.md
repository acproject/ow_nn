# 向量切片功能使用指南

## 概述

`ow_nn` 库现在支持对 `std::vector` 进行 Python 风格的切片操作，特别是针对 `shape` 向量的常用操作。

## 功能特性

### 1. 通用向量切片函数 `slice_vector`

```cpp
#include "src/utils.hpp"

template<typename T>
std::vector<T> slice_vector(const std::vector<T>& vec, int start, int end, int step = 1);
```

**特性：**
- 支持负数索引（-1 表示最后一个元素）
- 支持步长参数
- 自动边界检查
- 与 Python 切片语法兼容

**示例：**
```cpp
std::vector<int> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// vec[:-1] - 去掉最后一个元素
auto result1 = slice_vector(vec, 0, -1);
// 结果: {0, 1, 2, 3, 4, 5, 6, 7, 8}

// vec[1:-1] - 去掉首尾元素
auto result2 = slice_vector(vec, 1, -1);
// 结果: {1, 2, 3, 4, 5, 6, 7, 8}

// vec[::2] - 每隔一个元素取一个
auto result3 = slice_vector(vec, 0, vec.size(), 2);
// 结果: {0, 2, 4, 6, 8}
```

### 2. Shape 向量专用工具函数

在 `shape_utils` 命名空间下提供了针对 `shape` 向量的便捷函数：

```cpp
namespace shape_utils {
    // 获取除最后一维外的所有维度 (等价于 shape[:-1])
    std::vector<int> all_but_last(const std::vector<int>& shape);
    
    // 获取除第一维外的所有维度 (等价于 shape[1:])
    std::vector<int> all_but_first(const std::vector<int>& shape);
    
    // 获取最后n维 (等价于 shape[-n:])
    std::vector<int> last_n_dims(const std::vector<int>& shape, int n);
    
    // 获取前n维 (等价于 shape[:n])
    std::vector<int> first_n_dims(const std::vector<int>& shape, int n);
    
    // 获取中间维度 (等价于 shape[start:end])
    std::vector<int> middle_dims(const std::vector<int>& shape, int start, int end);
    
    // 获取最后一维的大小
    int last_dim(const std::vector<int>& shape);
    
    // 获取第一维的大小
    int first_dim(const std::vector<int>& shape);
}
```

## 实际应用示例

### 1. Attention 机制中的 input_shape 计算

```cpp
// 在 Qwen3Attention 中
std::vector<int> hidden_states_shape = {seq_len, hidden_size};

// Python: input_shape = hidden_states.shape[:-1]
auto input_shape = shape_utils::all_but_last(hidden_states_shape);
// 结果: {seq_len}
```

### 2. 卷积操作中的空间维度提取

```cpp
// 4D 卷积输出: [N, C, H, W]
std::vector<int> conv_shape = {1, 64, 56, 56};

// Python: spatial_size = shape[-2:]
auto spatial_size = shape_utils::last_n_dims(conv_shape, 2);
// 结果: {56, 56}

// Python: batch_channel = shape[:2]
auto batch_channel = shape_utils::first_n_dims(conv_shape, 2);
// 结果: {1, 64}
```

### 3. Reshape 操作

```cpp
std::vector<int> tensor_shape = {2, 3, 4, 5};

// 获取除最后一维外的所有维度
auto prefix_shape = shape_utils::all_but_last(tensor_shape);
// 结果: {2, 3, 4}

// 获取最后一维
int last_dim = shape_utils::last_dim(tensor_shape);
// 结果: 5

// 可以用于 reshape: [2, 3, 4] x 5 = [2, 3, 20]
```

### 4. 多头注意力中的维度处理

```cpp
// 原始 shape: [batch_size, seq_len, hidden_size]
std::vector<int> input_shape = {8, 512, 768};

int batch_size = shape_utils::first_dim(input_shape);
int seq_len = input_shape[1];
int hidden_size = shape_utils::last_dim(input_shape);

// 计算多头注意力的 shape
int num_heads = 12;
int head_dim = hidden_size / num_heads;

// 新的 shape: [batch_size, seq_len, num_heads, head_dim]
std::vector<int> multi_head_shape = {batch_size, seq_len, num_heads, head_dim};
```

## 与 Tensor 切片的区别

- **向量切片**：用于处理 `std::vector<int>` 类型的 shape 信息
- **Tensor 切片**：用于处理 `Tensor` 对象本身的数据切片

```cpp
// 向量切片 - 处理 shape 信息
std::vector<int> shape = tensor->shape;
auto input_shape = shape_utils::all_but_last(shape);

// Tensor 切片 - 处理 tensor 数据
auto sliced_tensor = tensor->slice_view(0, 0, seq_len);
```

## 性能说明

- 所有切片操作都会创建新的向量（拷贝数据）
- 对于频繁的 shape 操作，建议缓存结果
- `shape_utils` 函数针对常用模式进行了优化

## 编译和测试

运行测试程序：
```bash
g++ -std=c++17 test_vector_slice.cpp -o test_vector_slice
./test_vector_slice
```

项目编译：
```bash
cmake -B build && cmake --build build
```

## 注意事项

1. 负数索引会自动转换为正数索引
2. 越界访问会被自动修正到有效范围
3. 步长必须为正数
4. 空向量的切片操作会返回空向量