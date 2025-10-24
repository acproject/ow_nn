#pragma once

#include "ops.hpp"
#include "tensor.hpp"
#include <tuple>

namespace ow::nn {
// 参考 Python: x1 = x[..., :L/2]; x2 = x[..., L/2:]; cat((-x2, x1), dim=-1)
static inline TensorPtr rotate_half(const TensorPtr &x) {
  if (!x)
    throw std::runtime_error("rotate_half: input is null");
  int r = (int)x->shape.size();
  if (r <= 0)
    throw std::runtime_error("rotate_half: rank must be >= 1");
  int last = r - 1;
  int L = x->shape[last];
  if (L <= 0)
    throw std::runtime_error("rotate_half: last dim must be > 0");
  if ((L % 2) != 0)
    throw std::runtime_error("rotate_half: last dim must be even");
  int half = L / 2;

  // x1 = x[..., :half]
  std::vector<int> starts1(r, 0);
  std::vector<int> lens1 = x->shape;
  lens1[last] = half;
  auto x1 = x->slice_view(starts1, lens1);

  // x2 = x[..., half:]
  std::vector<int> starts2(r, 0);
  starts2[last] = half;
  std::vector<int> lens2 = x->shape;
  lens2[last] = L - half;
  auto x2 = x->slice_view(starts2, lens2);

  // -x2
  auto nx2 = x2->elementwise_unary(x2, [](float v) { return -v; });

  // concat([-x2, x1], dim=-1)
  return concat({nx2, x1}, -1);
}

/**
 * @brief 对 query 和 key 张量应用旋转位置编码（RoPE）。
 *
 * @param q 查询张量。
 * @param k 键张量。
 * @param cos 旋转嵌入的余弦部分。
 * @param sin 旋转嵌入的正弦部分。
 * @param position_ids 已弃用且未使用。
 * @param unsqueeze_dim 指定沿哪个维度对 cos[position_ids] 和 sin[position_ids]
 * 进行 unsqueeze， 以便它们能与 q 和 k 的维度正确广播。例如，cos[position_ids]
 * 与 sin[position_ids] 形状为 [batch_size, seq_len, head_dim]，若 q 和 k 形状为
 *                      [batch_size, heads, seq_len, head_dim]，则设
 * unsqueeze_dim=1； 若 q 和 k 形状为 [batch_size, seq_len, heads,
 * head_dim]，则设 unsqueeze_dim=2。
 * @return std::tuple<TensorPtr> 包含经过旋转位置编码后的 query 和 key 张量。
 */
static inline std::tuple<TensorPtr, TensorPtr>
apply_rotary_pos_emb(const TensorPtr &q, const TensorPtr &k,
                     const TensorPtr &cos, const TensorPtr &sin,
                     const TensorPtr &position_ids, int unsqueeze_dim) {
  (void)position_ids; // 未使用，但保留签名以兼容上层接口
  if (!q || !k || !cos || !sin)
    throw std::runtime_error("apply_rotary_pos_emb: null input");

  // 对 cos/sin 在指定维度 unsqueeze 以便与 q/k 广播
  auto cos_u = unsqueeze(cos, unsqueeze_dim);
  auto sin_u = unsqueeze(sin, unsqueeze_dim);

  // q_embed = (q * cos) + (rotate_half(q) * sin)
  auto q_mul = q->tensor_mul(q, cos_u);
  auto q_rot = rotate_half(q);
  auto q_rot_mul = q_rot->tensor_mul(q_rot, sin_u);
  auto q_embed = q_mul->tensor_add(q_mul, q_rot_mul);

  // k_embed = (k * cos) + (rotate_half(k) * sin)
  auto k_mul = k->tensor_mul(k, cos_u);
  auto k_rot = rotate_half(k);
  auto k_rot_mul = k_rot->tensor_mul(k_rot, sin_u);
  auto k_embed = k_mul->tensor_add(k_mul, k_rot_mul);

  return std::make_tuple(q_embed, k_embed);
}

// -------------------- 向量切片工具函数 --------------------

/**
 * @brief 对std::vector进行Python风格的切片操作
 * 
 * @tparam T 向量元素类型
 * @param vec 输入向量
 * @param start 起始索引（支持负数，-1表示最后一个元素）
 * @param end 结束索引（不包含，支持负数）
 * @param step 步长（默认为1）
 * @return std::vector<T> 切片后的向量
 * 
 * 使用示例：
 * - slice_vector(vec, 0, -1)  // 等价于 vec[:-1]，去掉最后一个元素
 * - slice_vector(vec, 1, -1)  // 等价于 vec[1:-1]，去掉首尾元素
 * - slice_vector(vec, -3, -1) // 等价于 vec[-3:-1]，倒数第3到倒数第2个元素
 * - slice_vector(vec, 0, vec.size(), 2) // 等价于 vec[::2]，每隔一个元素取一个
 */
template<typename T>
std::vector<T> slice_vector(const std::vector<T>& vec, int start, int end, int step = 1) {
    if (vec.empty()) return {};
    if (step <= 0) throw std::runtime_error("slice_vector: step must be positive");
    
    int size = static_cast<int>(vec.size());
    
    // 处理负数索引
    if (start < 0) start += size;
    if (end < 0) end += size;
    
    // 边界检查和调整
    start = std::max(0, std::min(start, size));
    end = std::max(0, std::min(end, size));
    
    if (start >= end) return {};
    
    std::vector<T> result;
    for (int i = start; i < end; i += step) {
        result.push_back(vec[i]);
    }
    
    return result;
}

/**
 * @brief 专门用于shape向量的切片函数，提供常用的切片操作
 */
namespace shape_utils {
    
    /**
     * @brief 获取除最后一维外的所有维度 (等价于 shape[:-1])
     */
    inline std::vector<int> all_but_last(const std::vector<int>& shape) {
        return slice_vector(shape, 0, -1);
    }
    
    /**
     * @brief 获取除第一维外的所有维度 (等价于 shape[1:])
     */
    inline std::vector<int> all_but_first(const std::vector<int>& shape) {
        return slice_vector(shape, 1, static_cast<int>(shape.size()));
    }
    
    /**
     * @brief 获取最后n维 (等价于 shape[-n:])
     */
    inline std::vector<int> last_n_dims(const std::vector<int>& shape, int n) {
        return slice_vector(shape, -n, static_cast<int>(shape.size()));
    }
    
    /**
     * @brief 获取前n维 (等价于 shape[:n])
     */
    inline std::vector<int> first_n_dims(const std::vector<int>& shape, int n) {
        return slice_vector(shape, 0, n);
    }
    
    /**
     * @brief 获取中间的维度 (等价于 shape[start:end])
     */
    inline std::vector<int> middle_dims(const std::vector<int>& shape, int start, int end) {
        return slice_vector(shape, start, end);
    }
    
    /**
     * @brief 获取最后一个维度的大小
     */
    inline int last_dim(const std::vector<int>& shape) {
        return shape.empty() ? 0 : shape.back();
    }
    
    /**
     * @brief 获取第一个维度的大小
     */
    inline int first_dim(const std::vector<int>& shape) {
        return shape.empty() ? 0 : shape.front();
    }
}

} // namespace ow::nn