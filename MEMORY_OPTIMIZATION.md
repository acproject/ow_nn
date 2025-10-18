# 内存优化改进

## 问题描述

原始程序在加载大型模型时遇到严重的内存分配问题：

1. **Context分配失败**: 尝试分配超过22GB内存失败
2. **权重加载跳过**: 由于内存不足导致权重加载失败
3. **推理错误**: 缺失权重导致模型推理失败

## 解决方案

### 1. Context类内存管理优化 (`src/context.hpp`)

**改进内容:**
- **保守的增长策略**: 根据当前arena大小采用不同的增长策略
- **系统内存检查**: 在分配前检查可用系统内存
- **安全限制**: 限制单次分配不超过可用内存的50%
- **更好的错误处理**: 提供详细的内存使用信息

**关键改进:**
```cpp
// 检查可用系统内存
size_t available_memory = get_available_memory();
size_t max_safe_allocation = available_memory / 2;

// 根据arena大小采用不同策略
if (current_size > 12GB) {
    // 超大arena: 仅分配所需 + 256MB缓冲
    new_size = needed_size + 256MB;
} else if (current_size > 4GB) {
    // 大arena: 保守增长
    new_size = min(needed_size + 1GB, current_size + current_size/4);
} else {
    // 小arena: 适度增长
    new_size = max(current_size + current_size/2, needed_size);
}
```

### 2. 智能权重加载器 (`src/memory_optimizer.hpp`)

**新功能:**
- **自适应初始大小**: 根据系统内存自动确定最优Context大小
- **分批加载**: 将权重分批加载以避免内存峰值
- **内存监控**: 实时监控内存使用情况
- **优先级加载**: 优先加载关键权重

**使用示例:**
```cpp
// 自动确定最优Context大小
size_t optimal_size = MemoryOptimizer::get_optimal_initial_context_size();
auto ctx = std::make_shared<Context>(optimal_size);

// 智能权重加载
auto weights = MemoryOptimizer::load_weights_optimized(
    loader, ctx, weight_patterns, true // 使用内存映射
);
```

### 3. 优化的权重加载策略 (`main.cpp`)

**改进内容:**
- **内存映射优先**: 使用内存映射减少内存拷贝
- **分类加载**: 区分关键权重和层权重
- **渐进式加载**: 分批加载大型transformer层
- **内存监控**: 在关键节点监控内存使用

## 内存使用策略

### 系统内存分配策略

| 可用内存 | 初始Context大小 | 策略 |
|---------|----------------|------|
| < 8GB   | 512MB          | 保守模式 |
| 8-16GB  | 1GB            | 标准模式 |
| 16-32GB | 2GB            | 高性能模式 |
| > 32GB  | 4GB            | 最大性能模式 |

### 权重加载优先级

1. **关键权重** (优先加载):
   - `model.embed_tokens.weight`
   - `model.norm.weight`
   - `lm_head.weight`

2. **层权重** (分批加载):
   - `model.language_model.layers.*`
   - 每批10层，监控内存使用

## 使用建议

### 对于低内存系统 (< 16GB)
- 程序会自动使用保守的内存分配策略
- 建议关闭其他大内存应用程序
- 考虑使用模型量化版本

### 对于高内存系统 (> 32GB)
- 程序会使用更大的初始Context大小
- 可以加载更大的模型
- 支持更高的批处理大小

## 监控和调试

程序现在提供详细的内存使用信息：

```
[Memory] Initial: Total: 32.0 GB, Used: 8.5 GB, Available: 23.5 GB
[MemOpt] Loading 4 essential weights...
[Memory] After essential weights: Available: 22.1 GB
[MemOpt] Loading 1247 layer weights in batches...
[Memory] Final: Available: 18.3 GB
```

## 测试

运行内存优化测试：
```bash
g++ -std=c++17 -I. test_memory_opt.cpp -o test_memory_opt
./test_memory_opt
```

## 故障排除

### 如果仍然遇到内存问题：

1. **检查系统内存**: 确保有足够的可用RAM
2. **关闭其他应用**: 释放更多内存给模型加载
3. **使用更小的模型**: 考虑使用量化或更小的模型版本
4. **增加虚拟内存**: 在Windows中增加页面文件大小

### 常见错误信息：

- `Single tensor allocation too large`: 单个张量超过8GB，考虑模型分片
- `Cannot allocate X GB. Available memory: Y GB`: 系统内存不足
- `High memory usage detected`: 内存使用超过70%，程序会自动调整策略

这些优化应该显著改善大型模型的加载成功率和内存使用效率。