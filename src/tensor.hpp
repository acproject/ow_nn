#pragma once
#include "../include/common.h"
#include "context.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>

// 前缀声明，用于定义下面的智能指针
namespace ow::nn {
struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
struct QuantParams {
  float scale = 1.0f;
};

// 轻量描述信息，用于图的形状与类型推断
struct TensorDesc {
  DType dtype;
  std::vector<int> shape;
  std::vector<int> strides;
};

struct Tensor : public std::enable_shared_from_this<Tensor> {
  DType dtype;
  std::vector<int> shape;
  std::vector<int> strides;
  void *data = nullptr;
  std::weak_ptr<Context> ctx; // 通过弱指针引用避免循环引用
  QuantParams qparams;

  Tensor(DType dtype, const std::vector<int> &shape)
      : dtype(dtype), shape(shape) {
    strides = calc_strides(shape);
  }

  // 根据给定的 shape 计算张量的 strides（步长）。
  // 结果 strides[i] 表示：沿着第 i
  // 维移动一个元素，在底层一维内存中需要跳过多少个元素。
  // 采用行主序（row-major）规则，即从最后一维开始反向累乘。
  // 示例：shape = {2, 3, 4}，则返回 strides = {12, 4, 1}。
  static std::vector<int> calc_strides(const std::vector<int> &shape) {
    std::vector<int> s(shape.size());
    int acc = 1;
    // 从最后一维开始反向遍历，逐步累乘
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      s[i] = acc;      // 当前维的步长 = 之前所有维长度的乘积
      acc *= shape[i]; // 更新累乘值，供前一维使用
    }
    return s;
  }

public:
  size_t nelements() const {
    size_t n = 1;
    for (int d : shape)
      n *= (size_t)d;
    return n;
  }

  // 计算当前 Tensor 占用的总字节数
  // 返回值 = 元素总数 × 单个元素的字节大小
  size_t nbytes() const {
    if (dtype == DType::Q4_0)
      return (nelements() + 1) / 2;
    return nelements() * dtype_size(dtype);
  }

  // 获取描述信息
  TensorDesc desc() const { return TensorDesc{dtype, shape, strides}; }

  // Create tensor backed by Context arena
  static TensorPtr create(const std::shared_ptr<Context> &ctx,
                          const std::vector<int> &shape,
                          DType dtype = DType::FLOAT32) {
    auto t = std::shared_ptr<Tensor>(new Tensor(dtype, shape));
    t->ctx = ctx;
    // 如果 ctx 有效，则通过其 arena 分配内存并清零
    if (auto c = ctx) {
      t->data = c->alloc(t->nbytes());
      std::memset(t->data, 0, t->nbytes()); // 将分配的内存初始化为0
      if (dtype == DType::Q4_0)
        t->qparams.scale = 1.0f;
    } else {
      throw std::runtime_error("Context expired when creating tensor");
    }
    return t;
  }

  bool is_contiguous_row_major() const {
    return strides == calc_strides(shape);
  }

  std::shared_ptr<Tensor> reshape_view(const std::vector<int> &new_shape) {
    size_t new_ne = 1;
    for (int d : new_shape)
      new_ne *= (size_t)d;
    if (new_ne != nelements()) {
      std::cerr << "[Tensor] reshape_view size mismatch: old_shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i) std::cerr << ",";
        std::cerr << shape[i];
      }
      std::cerr << "] ne=" << nelements() << " new_shape=[";
      for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i) std::cerr << ",";
        std::cerr << new_shape[i];
      }
      std::cerr << "] new_ne=" << new_ne << std::endl;
      throw std::runtime_error("reshape size mismatch");
    }
    if (!is_contiguous_row_major()) {
      std::cerr << "[Tensor] reshape_view requires contiguous tensor: old_shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i) std::cerr << ",";
        std::cerr << shape[i];
      }
      std::cerr << "] strides=[";
      for (size_t i = 0; i < strides.size(); ++i) {
        if (i) std::cerr << ",";
        std::cerr << strides[i];
      }
      std::cerr << "]" << std::endl;
      throw std::runtime_error("reshape_view requires contiguous tensor");
    }
    Tensor *raw = new Tensor(dtype, new_shape);
    raw->strides = calc_strides(new_shape);
    raw->data = data;
    raw->ctx = ctx;
    return std::shared_ptr<Tensor>(shared_from_this(), raw);
  }

  std::shared_ptr<Tensor> transpose_view(int axis0 = 0, int axis1 = 1) {
    if (shape.size() < 2)
      throw std::runtime_error("transpose needs rank>=2");
    std::vector<int> ns = shape;
    std::swap(ns[axis0], ns[axis1]);
    std::vector<int> nstr = strides;
    std::swap(nstr[axis0], nstr[axis1]);
    Tensor *raw = new Tensor(dtype, ns);
    raw->strides = nstr;
    raw->data = data;
    raw->ctx = ctx;
    return std::shared_ptr<Tensor>(shared_from_this(), raw);
  }

  std::shared_ptr<Tensor> slice_view(const std::vector<int> &starts,
                                     const std::vector<int> &lengths) {
    if (starts.size() != shape.size() || lengths.size() != shape.size())
      throw std::runtime_error("slice dims mismatch");
    std::vector<int> new_shape(shape.size());
    size_t elem_offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
      int s = starts[i];
      int len = lengths[i];
      if (s < 0 || s >= shape[i])
        throw std::runtime_error("slice start out of range");
      if (len == -1)
        len = shape[i] - s;
      if (len <= 0 || s + len > shape[i])
        throw std::runtime_error("slice len out of range");
      new_shape[i] = len;
      elem_offset += (size_t)s * (size_t)strides[i];
    }
    Tensor *raw = new Tensor(dtype, new_shape);
    raw->strides = strides;
    raw->data = static_cast<uint8_t *>(data) + elem_offset * dtype_size(dtype);
    raw->ctx = ctx;
    return std::shared_ptr<Tensor>(shared_from_this(), raw);
  }

  // Create a view (shared data pointer, no copy)
  TensorPtr view(const std::vector<int> &new_shape,
                 const std::vector<int> &new_strides, size_t offset_bytes = 0) {
    auto self = shared_from_this();
    Tensor *raw_view = new Tensor(dtype, new_shape);
    raw_view->strides = new_strides;
    raw_view->data = static_cast<uint8_t *>(data) + offset_bytes;
    raw_view->ctx = ctx;
    // share same lifetime as self
    return TensorPtr(self, raw_view);
  }

  // 创建一个复制副本（保持 dtype 与布局），数据拷贝
  TensorPtr copy() const {
    auto c = ctx.lock();
    if (!c) throw std::runtime_error("ctx expired");
    auto out = Tensor::create(c, shape, dtype);
    std::memcpy(out->data, data, nbytes());
    return out;
  }

  // 转换 dtype，逐元素转换
  TensorPtr astype(DType new_dtype) const {
    if (new_dtype == dtype) return copy();
    auto c = ctx.lock();
    if (!c) throw std::runtime_error("ctx expired");
    auto out = Tensor::create(c, shape, new_dtype);
    size_t n = nelements();
    for (size_t i = 0; i < n; ++i) {
      float v = get_as_float_flat(i);
      out->set_from_float_flat(i, v);
    }
    return out;
  }

  // 以 float& 形式返回第 idx 个元素，仅当 dtype 为 FLOAT32 时可用
  float &at_f(size_t idx) {
    assert(dtype == DType::FLOAT32); // 运行时断言：确保数据类型匹配
    return reinterpret_cast<float *>(
        data)[idx]; // 将裸指针强转为 float* 后取下标
  }

  const float &at_f(size_t idx) const {
    assert(dtype == DType::FLOAT32);
    return reinterpret_cast<const float *>(data)[idx];
  }

  // ----------------------- quantization convert utilities
  // ------------------------- Convert contigiuos float array -> int8 storage in
  // Tensor
  static void quantize_int8_from_floats(const TensorPtr &T_dst,
                                        const std::vector<float> &src) {
    assert(T_dst->dtype == DType::INT8);
    size_t n = T_dst->nelements();
    if (src.size() != n)
      throw std::runtime_error("quantize_int8: size");
    for (size_t i = 0; i < n; ++i) {
      T_dst->set_from_float_flat(i, src[i]);
    }
  }

  // flattened access helpers
  // FP8 decode helpers (E4M3, E5M2)
  static inline float fp8_e4m3_to_f32(uint8_t x) {
    uint8_t sign = (x >> 7) & 0x1;
    uint8_t exp = (x >> 3) & 0x0F;
    uint8_t mant = x & 0x07;
    const int bias = 7;
    if (exp == 0) {
      if (mant == 0) return sign ? -0.0f : 0.0f;
      float m = mant / 8.0f;
      float val = std::ldexp(m, 1 - bias);
      return sign ? -val : val;
    } else if (exp == 0x0F) {
      // Inf / NaN
      if (mant == 0) return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      float m = 1.0f + mant / 8.0f;
      int E = (int)exp - bias;
      float val = std::ldexp(m, E);
      return sign ? -val : val;
    }
  }
  static inline float fp8_e5m2_to_f32(uint8_t x) {
    uint8_t sign = (x >> 7) & 0x1;
    uint8_t exp = (x >> 2) & 0x1F;
    uint8_t mant = x & 0x03;
    const int bias = 15;
    if (exp == 0) {
      if (mant == 0) return sign ? -0.0f : 0.0f;
      float m = mant / 4.0f;
      float val = std::ldexp(m, 1 - bias);
      return sign ? -val : val;
    } else if (exp == 0x1F) {
      // Inf / NaN
      if (mant == 0) return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      float m = 1.0f + mant / 4.0f;
      int E = (int)exp - bias;
      float val = std::ldexp(m, E);
      return sign ? -val : val;
    }
  }

  float get_as_float_flat(size_t index) const {
    if (dtype == DType::FLOAT32) {
      const float *p = reinterpret_cast<const float *>(data);
      return p[index];
    }
    if (dtype == DType::FP16) {
      const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
      return fp16_to_float(p[index]);
    }
    if (dtype == DType::BF16) {
      const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
      return bf16_to_float(p[index]);
    }
    if (dtype == DType::INT32) {
      const int32_t *p = reinterpret_cast<const int32_t *>(data);
      return float(p[index]);
    }
    if (dtype == DType::INT8) {
      const int8_t *p = reinterpret_cast<const int8_t *>(data);
      return float(p[index]);
    }
    if (dtype == DType::U8) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
      return float(p[index]);
    }
    if (dtype == DType::BOOL) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
      return p[index] ? 1.0f : 0.0f;
    }
    if (dtype == DType::I16) {
      const int16_t *p = reinterpret_cast<const int16_t *>(data);
      return float(p[index]);
    }
    if (dtype == DType::I64) {
      const int64_t *p = reinterpret_cast<const int64_t *>(data);
      return float(p[index]);
    }
    if (dtype == DType::F64) {
      const double *p = reinterpret_cast<const double *>(data);
      return (float)p[index];
    }
    if (dtype == DType::FP8_E4M3) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
      return fp8_e4m3_to_f32(p[index]);
    }
    if (dtype == DType::FP8_E5M2) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
      return fp8_e5m2_to_f32(p[index]);
    }
    if (dtype == DType::Q4_0) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
      size_t byte_idx = index / 2;
      bool high = (index % 2) == 0;
      uint8_t b = p[byte_idx];
      int8_t q = high ? ((b >> 4) & 0xF) : (b & 0xF);
      if (q & 0x8)
        q = q - 16;
      return float(q) * qparams.scale;
    }
    return 0.0f;
  }

  void set_from_float_flat(size_t index, float v) {
    if (dtype == DType::FLOAT32) {
      float *p = reinterpret_cast<float *>(data);
      p[index] = v;
      return;
    }
    if (dtype == DType::FP16) {
      uint16_t *p = reinterpret_cast<uint16_t *>(data);
      p[index] = float_to_fp16(v);
      return;
    }
    if (dtype == DType::BF16) {
      uint16_t *p = reinterpret_cast<uint16_t *>(data);
      p[index] = float_to_bf16(v);
      return;
    }
    if (dtype == DType::INT32) {
      int32_t *p = reinterpret_cast<int32_t *>(data);
      p[index] = float(v);
      return;
    }
    if (dtype == DType::INT8) {
      int8_t *p = reinterpret_cast<int8_t *>(data);
      int32_t vv = (int32_t)std::round(v);
      if (vv > 127)
        vv = 127;
      if (vv < -128)
        vv = -128;
      p[index] = (int8_t)vv;
      return;
    }
    if (dtype == DType::Q4_0) {
      uint8_t *p = reinterpret_cast<uint8_t *>(data);
      int32_t q = (int32_t)std::round(v / qparams.scale);
      if (q > 7)
        q = 7;
      if (q < -8)
        q = -8;
      uint8_t uq = (uint8_t)(q & 0xF);
      size_t byte_idx = index / 2;
      bool high = (index % 2) == 0;
      uint8_t old = p[byte_idx];
      if (high)
        old = (old & 0x0F) | (uq << 4);
      else
        old = (old & 0xF0) | (uq & 0x0F);
      p[byte_idx] = old;
      return;
    }
  }

  // ----------------------- broadcasting helpers -----------------------
  static std::vector<int> broadcast_shape(const std::vector<int>& a,
                                          const std::vector<int>& b) {
    size_t ra = a.size(), rb = b.size();
    size_t r = std::max(ra, rb);
    std::vector<int> out(r, 1);
    for (size_t i = 0; i < r; ++i) {
      int ad = (i < r - ra) ? 1 : a[i - (r - ra)];
      int bd = (i < r - rb) ? 1 : b[i - (r - rb)];
      if (ad != bd && ad != 1 && bd != 1)
        throw std::runtime_error("broadcast: incompatible dims");
      out[i] = std::max(ad, bd);
    }
    return out;
  }

  static size_t linear_index(const std::vector<int>& shape,
                             const std::vector<int>& idx) {
    size_t r = shape.size();
    size_t off = 0;
    for (size_t i = 0; i < r; ++i) off = off * (size_t)shape[i] + (size_t)idx[i];
    return off;
  }

  static void next_index(std::vector<int>& idx,
                         const std::vector<int>& shape) {
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      idx[i]++;
      if (idx[i] < shape[i]) return;
      idx[i] = 0;
    }
  }

  // Q4_0: compute scale and pack
  static void quantize_q4_0_from_floats(const TensorPtr &T_dst,
                                        const std::vector<float> &src) {
    assert(T_dst->dtype == DType::Q4_0);
    size_t n = T_dst->nelements();
    if (src.size() != n)
      throw std::runtime_error(
          "quantize_q4_0: size"); // choose scale as max(abs)/ 7
    float maxabs = 0.0f;
    for (float v : src)
      maxabs = std::max(maxabs, std::fabs(v));
    if (maxabs < 1e-8f)
      maxabs = 1.0f;
    T_dst->qparams.scale = maxabs / 7.0f;
    for (size_t i = 0; i < n; ++i)
      T_dst->set_from_float_flat(i, src[i]);
  }

  // Dequantize to float vector
  std::vector<float> dequantize_to_floats(const TensorPtr &T) {
    size_t n = T->nelements();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
      out[i] = T->get_as_float_flat(i);
    return out;
  }

  void print_f(const std::string &name = "") {
    std::cout << (name.empty() ? "Tensor" : name) << " dtype=" << (int)dtype
              << " shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << shape[i];
    }
    std::cout << "]\n";
    size_t n = nelements();
    for (size_t i = 0; i < n; ++i) {
      float v = get_as_float_flat(i);
      std::cout << std::fixed << std::setprecision(4) << v << " ";
    }
    std::cout << "\n";
  }

// -------------------------- blocked, cache-friendly mulithreaded matmul
// --------- implement a simple blocking algorithm with outer parallelism over
// blocks of rows of A.
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
  // SIMD pack helper: pack B panel [p0,p1) x [j0,j1) into contiguous buffer
  // Uses scratch buffer for better memory efficiency
  static inline float* ow_pack_B_panel_scratch(const TensorPtr &B, int j0, int j1, int p0,
                                               int p1, int W, const std::shared_ptr<Context>& ctx) {
    int k_len = p1 - p0;
    float* pack = static_cast<float*>(ctx->scratch_alloc_matrix_pack(k_len, W));
    int n = B->shape[1];
    for (int p = 0; p < k_len; ++p) {
      int src_p = p0 + p;
      for (int l = 0; l < W; ++l) {
        int j = j0 + l;
        float bv = 0.0f;
        if (j < j1) {
          size_t bi = (size_t)src_p * n + j;
          bv = B->get_as_float_flat(bi);
          if (!std::isfinite(bv)) bv = 0.0f;
          else if (bv > 1e6f) bv = 1e6f;
          else if (bv < -1e6f) bv = -1e6f;
        }
        pack[p * W + l] = bv;
      }
    }
    return pack;
  }

  // Legacy vector-based pack helper (kept for compatibility)
  static inline void ow_pack_B_panel(const TensorPtr &B, int j0, int j1, int p0,
                                     int p1, int W, std::vector<float> &pack) {
    int k_len = p1 - p0;
    pack.resize(k_len * W);
    int n = B->shape[1];
    for (int p = 0; p < k_len; ++p) {
      int src_p = p0 + p;
      for (int l = 0; l < W; ++l) {
        int j = j0 + l;
        float bv = 0.0f;
        if (j < j1) {
          size_t bi = (size_t)src_p * n + j;
          bv = B->get_as_float_flat(bi);
          if (!std::isfinite(bv)) bv = 0.0f;
          else if (bv > 1e6f) bv = 1e6f;
          else if (bv < -1e6f) bv = -1e6f;
        }
        pack[p * W + l] = bv;
      }
    }
  }

  // Transposed pack helper with scratch buffer: treat B with shape [n, k] as B^T when packing
  static inline float* ow_pack_B_panel_transposed_scratch(const TensorPtr &B, int j0, int j1, int p0,
                                                          int p1, int W, const std::shared_ptr<Context>& ctx) {
    int k_len = p1 - p0;
    float* pack = static_cast<float*>(ctx->scratch_alloc_matrix_pack(k_len, W));
    int k = B->shape[1];
    for (int p = 0; p < k_len; ++p) {
      int src_p = p0 + p; // along original k dimension
      for (int l = 0; l < W; ++l) {
        int j = j0 + l;   // along original n dimension
        float bv = 0.0f;
        if (j < j1) {
          size_t bi = (size_t)j * k + src_p; // index [j, src_p]
          bv = B->get_as_float_flat(bi);
          if (!std::isfinite(bv)) bv = 0.0f;
          else if (bv > 1e6f) bv = 1e6f;
          else if (bv < -1e6f) bv = -1e6f;
        }
        pack[p * W + l] = bv;
      }
    }
    return pack;
  }

  // Legacy transposed pack helper: treat B with shape [n, k] as B^T when packing
  static inline void ow_pack_B_panel_transposed(const TensorPtr &B, int j0, int j1, int p0,
                                     int p1, int W, std::vector<float> &pack) {
    int k_len = p1 - p0;
    pack.resize(k_len * W);
    int k = B->shape[1];
    for (int p = 0; p < k_len; ++p) {
      int src_p = p0 + p; // along original k dimension
      for (int l = 0; l < W; ++l) {
        int j = j0 + l;   // along original n dimension
        float bv = 0.0f;
        if (j < j1) {
          size_t bi = (size_t)j * k + src_p; // index [j, src_p]
          bv = B->get_as_float_flat(bi);
          if (!std::isfinite(bv)) bv = 0.0f;
          else if (bv > 1e6f) bv = 1e6f;
          else if (bv < -1e6f) bv = -1e6f;
        }
        pack[p * W + l] = bv;
      }
    }
  }

#if defined(__AVX__)
  // 8-wide AVX micro-kernel for one row with numerical stability
  static inline void ow_microkernel_row_8_avx(const float *Arow_p0,
                                              const float *packB, int k_len,
                                              float *Rptr) {
    __m256 acc = _mm256_setzero_ps();
    __m256 clamp_max = _mm256_set1_ps(1e6f);
    __m256 clamp_min = _mm256_set1_ps(-1e6f);
    
    for (int p = 0; p < k_len; ++p) {
      __m256 b = _mm256_loadu_ps(packB + p * 8);
      __m256 a = _mm256_set1_ps(Arow_p0[p]);
      
      // Clamp inputs to prevent overflow
      a = _mm256_max_ps(_mm256_min_ps(a, clamp_max), clamp_min);
      b = _mm256_max_ps(_mm256_min_ps(b, clamp_max), clamp_min);
      
#if defined(__FMA__)
      acc = _mm256_fmadd_ps(a, b, acc);
#else
      acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
      
      // Clamp accumulator to prevent overflow
      acc = _mm256_max_ps(_mm256_min_ps(acc, clamp_max), clamp_min);
    }
    
    __m256 prev = _mm256_loadu_ps(Rptr);
    prev = _mm256_add_ps(prev, acc);
    // Final clamp before storing
    prev = _mm256_max_ps(_mm256_min_ps(prev, clamp_max), clamp_min);
    _mm256_storeu_ps(Rptr, prev);
  }
#endif

  // 4-wide SSE micro-kernel for one row with numerical stability
#if defined(__SSE__)
  static inline void ow_microkernel_row_4_sse(const float *Arow_p0,
                                              const float *packB, int k_len,
                                              float *Rptr, int stride) {
    __m128 acc = _mm_setzero_ps();
    __m128 clamp_max = _mm_set1_ps(1e6f);
    __m128 clamp_min = _mm_set1_ps(-1e6f);
    
    for (int p = 0; p < k_len; ++p) {
      __m128 b = _mm_loadu_ps(packB + p * stride);
      __m128 a = _mm_set1_ps(Arow_p0[p]);
      
      // Clamp inputs to prevent overflow
      a = _mm_max_ps(_mm_min_ps(a, clamp_max), clamp_min);
      b = _mm_max_ps(_mm_min_ps(b, clamp_max), clamp_min);
      
      acc = _mm_add_ps(acc, _mm_mul_ps(a, b));
      
      // Clamp accumulator to prevent overflow
      acc = _mm_max_ps(_mm_min_ps(acc, clamp_max), clamp_min);
    }
    
    __m128 prev = _mm_loadu_ps(Rptr);
    prev = _mm_add_ps(prev, acc);
    // Final clamp before storing
    prev = _mm_max_ps(_mm_min_ps(prev, clamp_max), clamp_min);
    _mm_storeu_ps(Rptr, prev);
  }
#endif

  // Cache-friendly matrix multiplication with optimized blocking strategy
  static TensorPtr matmul_cache_friendly(const TensorPtr &A, const TensorPtr &B,
                                         size_t nthreads = 0) {
    if (A->shape.size() != 2 || B->shape.size() != 2)
      throw std::runtime_error("matmul: need 2D");
    int m = A->shape[0];
    int k = A->shape[1];

    bool B_is_k_by_n = (B->shape[0] == k);
    bool B_is_n_by_k = (B->shape[1] == k);
    if (!B_is_k_by_n && !B_is_n_by_k)
      throw std::runtime_error("matmul dim");
    int n = B_is_k_by_n ? B->shape[1] : B->shape[0];

    auto ctx = A->ctx.lock();
    if (!ctx)
      throw std::runtime_error("ctx expired");
    auto R = Tensor::create(ctx, {m, n}, DType::FLOAT32);

    // Adaptive block sizes based on cache hierarchy and matrix dimensions
    // L1: 32KB, L2: 256KB, L3: 8MB (typical values)
    size_t L1_size = 32 * 1024;
    size_t L2_size = 256 * 1024;
    
    // Calculate optimal block sizes for cache efficiency
    // For small matrices, use smaller blocks; for large matrices, use larger blocks
    size_t block_k, block_m, block_n;
    
    if (k <= 64 && m <= 64 && n <= 64) {
        // Very small matrices - use minimal blocking
        block_k = std::min(k, 32);
        block_m = std::min(m, 32);
        block_n = std::min(n, 32);
    } else if (k <= 512 && m <= 512 && n <= 512) {
        // Medium matrices - moderate blocking
        block_k = std::min(k, 64);
        block_m = std::min(m, 64);
        block_n = std::min(n, 64);
    } else {
        // Large matrices - aggressive blocking for cache efficiency
        block_k = std::min(k, 128);
        block_m = std::min(m, 128);
        block_n = std::min(n, 128);
    }
    
    // Ensure minimum block sizes for SIMD efficiency
    block_m = std::max(block_m, (size_t)8);
    block_n = std::max(block_n, (size_t)8);
    block_k = std::max(block_k, (size_t)8);
    
    if (const char* vv = std::getenv("OWNN_VERBOSE_MATMUL"); vv && vv[0] != '0') {
      std::cout << "[MatMul] Cache-friendly blocks: M=" << block_m 
                << ", N=" << block_n << ", K=" << block_k << std::endl;
    }

    if (nthreads == 0)
      nthreads = std::max<size_t>(1, std::thread::hardware_concurrency());
    ThreadPool tp(nthreads);

#if defined(__AVX__)
    const int W = 8;
#else
    const int W = 4;
#endif

    const bool fastA = (A->dtype == DType::FLOAT32);
    const float *Adata = fastA ? reinterpret_cast<const float *>(A->data) : nullptr;
    float *Rdata = reinterpret_cast<float *>(R->data);

    std::vector<std::future<void>> futs;
    for (int i0 = 0; i0 < m; i0 += (int)block_m) {
      int i1 = (std::min)(m, i0 + (int)block_m);
      futs.emplace_back(tp.submit([=, &A, &B, &R]() {
        // Reset scratch buffer at the start of each thread task
        auto ctx = A->ctx.lock();
        if (ctx) ctx->scratch_reset();
        
        for (int j0 = 0; j0 < n; j0 += (int)block_n) {
          int j1 = (std::min)(n, j0 + (int)block_n);
          for (int p0 = 0; p0 < k; p0 += (int)block_k) {
            int p1 = (std::min)(k, p0 + (int)block_k);
            for (int jpanel = j0; jpanel < j1; jpanel += W) {
              int jend = (std::min)(j1, jpanel + W);
              // Use scratch buffer for better memory efficiency
              float* pack;
              if (B_is_k_by_n) {
                pack = ow_pack_B_panel_scratch(B, jpanel, jend, p0, p1, W, ctx);
              } else {
                pack = ow_pack_B_panel_transposed_scratch(B, jpanel, jend, p0, p1, W, ctx);
              }
              int k_len = p1 - p0;
              for (int ii = i0; ii < i1; ++ii) {
                const float *Arow = fastA ? (Adata + ii * k + p0) : nullptr;
                float *Rptr = Rdata + (size_t)ii * n + jpanel;
#if defined(__AVX__)
                if (jend - jpanel == 8 && Arow) {
                  ow_microkernel_row_8_avx(Arow, pack, k_len, Rptr);
                  continue;
                }
#endif
#if defined(__SSE__)
                if (jend - jpanel == 4 && Arow) {
                  ow_microkernel_row_4_sse(Arow, pack, k_len, Rptr, W);
                  continue;
                }
#endif
                // fallback scalar for leftover columns or non-FLOAT32 A
                int width = jend - jpanel;
                float acc_buf[8];
                for (int l = 0; l < width; ++l) acc_buf[l] = 0.0f;
                
                for (int p = 0; p < k_len; ++p) {
                  float a = fastA ? Arow[p]
                                  : A->get_as_float_flat((size_t)ii * k + p0 + p);
                  if (!std::isfinite(a)) a = 0.0f;
                  const float *bvec = pack + p * W;
                  for (int l = 0; l < width; ++l) {
                    acc_buf[l] += a * bvec[l];
                  }
                }
                for (int l = 0; l < width; ++l) {
                  size_t ri = (size_t)ii * n + (jpanel + l);
                  float prev = R->get_as_float_flat(ri);
                  R->set_from_float_flat(ri, prev + acc_buf[l]);
                }
              }
              // Release scratch used by this panel to avoid accumulation
              if (ctx) ctx->scratch_reset();
            }
          }
        }
      }));
    }
    for (auto &f : futs)
      f.get();
    return R;
  }

  static TensorPtr matmul_blocked_mt(const TensorPtr &A, const TensorPtr &B,
                                     size_t block_m = 64, size_t block_n = 64,
                                     size_t block_k = 64, size_t nthreads = 0) {
    if (A->shape.size() != 2 || B->shape.size() != 2)
      throw std::runtime_error("matmul: need 2D");
    int m = A->shape[0];
    int k = A->shape[1];

    bool B_is_k_by_n = (B->shape[0] == k);
    bool B_is_n_by_k = (B->shape[1] == k);
    if (!B_is_k_by_n && !B_is_n_by_k)
      throw std::runtime_error("matmul dim");
    int n = B_is_k_by_n ? B->shape[1] : B->shape[0];

    auto ctx = A->ctx.lock();
    if (!ctx)
      throw std::runtime_error("ctx expired");
    auto R = Tensor::create(ctx, {m, n}, DType::FLOAT32);

    if (nthreads == 0)
      nthreads = std::max<size_t>(1, std::thread::hardware_concurrency());
    ThreadPool tp(nthreads);

#if defined(__AVX__)
    const int W = 8;
#else
    const int W = 4;
#endif

    const bool fastA = (A->dtype == DType::FLOAT32);
    const float *Adata =
        fastA ? reinterpret_cast<const float *>(A->data) : nullptr;
    float *Rdata = reinterpret_cast<float *>(R->data);

    std::vector<std::future<void>> futs;
    for (int i0 = 0; i0 < m; i0 += (int)block_m) {
      int i1 = (std::min)(m, i0 + (int)block_m);
      futs.emplace_back(tp.submit([=, &A, &B, &R]() {
        // Reset scratch buffer at the start of each thread task
        auto ctx = A->ctx.lock();
        if (ctx) ctx->scratch_reset();
        for (int j0 = 0; j0 < n; j0 += (int)block_n) {
          int j1 = (std::min)(n, j0 + (int)block_n);
          for (int p0 = 0; p0 < k; p0 += (int)block_k) {
            int p1 = (std::min)(k, p0 + (int)block_k);
            for (int jpanel = j0; jpanel < j1; jpanel += W) {
              int jend = (std::min)(j1, jpanel + W);
              // Use scratch buffer instead of vector allocation
              float* pack;
              if (B_is_k_by_n) {
                pack = ow_pack_B_panel_scratch(B, jpanel, jend, p0, p1, W, ctx);
              } else {
                pack = ow_pack_B_panel_transposed_scratch(B, jpanel, jend, p0, p1, W, ctx);
              }
              int k_len = p1 - p0;
              for (int ii = i0; ii < i1; ++ii) {
                const float *Arow = fastA ? (Adata + ii * k + p0) : nullptr;
                float *Rptr = Rdata + (size_t)ii * n + jpanel;
#if defined(__AVX__)
                if (jend - jpanel == 8 && Arow) {
                  ow_microkernel_row_8_avx(Arow, pack, k_len, Rptr);
                  continue;
                }
#endif
#if defined(__SSE__)
                if (jend - jpanel == 4 && Arow) {
                  ow_microkernel_row_4_sse(Arow, pack, k_len, Rptr, W);
                  continue;
                }
#endif
                // fallback scalar for leftover columns or non-FLOAT32 A
                int width = jend - jpanel;
                float acc_buf[8];
                for (int l = 0; l < width; ++l) acc_buf[l] = 0.0f;
                
                for (int p = 0; p < k_len; ++p) {
                  float a = fastA ? Arow[p]
                                  : A->get_as_float_flat((size_t)ii * k + p0 + p);
                  if (!std::isfinite(a)) a = 0.0f;
                  const float *bvec = pack + p * W;
                  for (int l = 0; l < width; ++l) {
                    acc_buf[l] += a * bvec[l];
                  }
                }
                for (int l = 0; l < width; ++l) {
                  size_t ri = (size_t)ii * n + (jpanel + l);
                  float prev = R->get_as_float_flat(ri);
                  R->set_from_float_flat(ri, prev + acc_buf[l]);
                }
              }
              // Release scratch used by this panel to avoid accumulation
              if (ctx) ctx->scratch_reset();
            }
          }
        }
      }));
    }
    for (auto &f : futs)
      f.get();
    return R;
  }

  // 专用 1xK · KxN 的并行 matvec（按列块并行），用于 LMHead 大词表场景
  // A: [1, K], B: [K, N] -> R: [1, N]
  static TensorPtr matvec_blocked_mt(const TensorPtr &A, const TensorPtr &B,
                                     size_t block_n = 4096,
                                     size_t block_k = 64,
                                     size_t nthreads = 0) {
    if (A->shape.size() != 2 || B->shape.size() != 2)
      throw std::runtime_error("matvec: need 2D");
    int m = A->shape[0];
    int k = A->shape[1];
    if (m != 1) throw std::runtime_error("matvec expects A as [1,K]");

    bool B_is_k_by_n = (B->shape[0] == k);
    bool B_is_n_by_k = (B->shape[1] == k);
    if (!B_is_k_by_n && !B_is_n_by_k) throw std::runtime_error("matvec dim");
    int n = B_is_k_by_n ? B->shape[1] : B->shape[0];

    auto ctx = A->ctx.lock();
    if (!ctx) throw std::runtime_error("ctx expired");
    auto R = Tensor::create(ctx, {1, n}, DType::FLOAT32);

    if (nthreads == 0)
      nthreads = std::max<size_t>(1, std::thread::hardware_concurrency());
    ThreadPool tp(nthreads);

#if defined(__AVX__)
    const int W = 8;
#else
    const int W = 4;
#endif

    const bool fastA = (A->dtype == DType::FLOAT32);
    const float *Adata = fastA ? reinterpret_cast<const float *>(A->data) : nullptr;
    float *Rdata = reinterpret_cast<float *>(R->data);

    // 初始化为 0（微核会做累加）
    for (int j = 0; j < n; ++j) Rdata[j] = 0.0f;

    std::vector<std::future<void>> futs;
    for (int j0 = 0; j0 < n; j0 += (int)block_n) {
      int j1 = (std::min)(n, j0 + (int)block_n);
      futs.emplace_back(tp.submit([=, &A, &B]() {
        // Reset scratch buffer at the start of each thread task
        auto ctx = A->ctx.lock();
        if (ctx) ctx->scratch_reset();
        
        for (int p0 = 0; p0 < k; p0 += (int)block_k) {
          int p1 = (std::min)(k, p0 + (int)block_k);
          const float *Arow_p0 = fastA ? (Adata + p0) : nullptr;
          int k_len = p1 - p0;
          for (int jpanel = j0; jpanel < j1; jpanel += W) {
            int jend = (std::min)(j1, jpanel + W);
            // Use scratch buffer instead of vector allocation
            float* pack;
            if (B_is_k_by_n) {
              pack = ow_pack_B_panel_scratch(B, jpanel, jend, p0, p1, W, ctx);
            } else {
              pack = ow_pack_B_panel_transposed_scratch(B, jpanel, jend, p0, p1, W, ctx);
            }
            float *Rptr = Rdata + jpanel;
#if defined(__AVX__)
            if (jend - jpanel == 8 && Arow_p0) {
              ow_microkernel_row_8_avx(Arow_p0, pack, k_len, Rptr);
              // Release scratch per panel
              if (ctx) ctx->scratch_reset();
              continue;
            }
#endif
#if defined(__SSE__)
            if (jend - jpanel == 4 && Arow_p0) {
              ow_microkernel_row_4_sse(Arow_p0, pack, k_len, Rptr, W);
              // Release scratch per panel
              if (ctx) ctx->scratch_reset();
              continue;
            }
#endif
            // fallback: 标量累加
            int width = jend - jpanel;
            for (int p = 0; p < k_len; ++p) {
              float a = fastA ? Arow_p0[p] : A->get_as_float_flat(p0 + p);
              if (!std::isfinite(a)) a = 0.0f;
              const float *bvec = pack + p * W;
              for (int l = 0; l < width; ++l) {
                Rptr[l] += a * bvec[l];
              }
            }
            // Release scratch per panel
            if (ctx) ctx->scratch_reset();
          }
        }
      }));
    }
    for (auto &f : futs) f.get();
    return R;
  }

  // ----------------------- elementwise ops (flattend)
  // ---------------------------
  TensorPtr elementwise_binary(const TensorPtr &A, const TensorPtr &B,
                               const std::function<float(float, float)> &op) {
    // 支持广播规则（尾维对齐，维度为1可扩展）
    auto ctx = A->ctx.lock();
    if (!ctx) throw std::runtime_error("ctx expired");
    auto out_shape = broadcast_shape(A->shape, B->shape);
    auto R = Tensor::create(ctx, out_shape, DType::FLOAT32);
    std::vector<int> idx(out_shape.size(), 0);
    size_t total = 1; for (int d: out_shape) total *= (size_t)d;
    for (size_t t = 0; t < total; ++t) {
      // map output idx to A/B idx
      std::vector<int> ia(A->shape.size(), 0), ib(B->shape.size(), 0);
      // align right (like numpy)
      size_t ra = A->shape.size(), rb = B->shape.size();
      for (size_t i = 0; i < out_shape.size(); ++i) {
        int coord = idx[i];
        if (i >= out_shape.size() - ra) {
          int ad = A->shape[i - (out_shape.size() - ra)];
          ia[i - (out_shape.size() - ra)] = (ad == 1) ? 0 : coord;
        }
        if (i >= out_shape.size() - rb) {
          int bd = B->shape[i - (out_shape.size() - rb)];
          ib[i - (out_shape.size() - rb)] = (bd == 1) ? 0 : coord;
        }
      }
      size_t la = linear_index(A->shape, ia);
      size_t lb = linear_index(B->shape, ib);
      float a = A->get_as_float_flat(la);
      float b = B->get_as_float_flat(lb);
      R->set_from_float_flat(t, op(a, b));
      next_index(idx, out_shape);
    }
    return R;
  }

  TensorPtr tensor_add(const TensorPtr &A, const TensorPtr &B) {
    return elementwise_binary(A, B, [](float a, float b) { return a + b; });
  }

  TensorPtr tensor_sub(const TensorPtr &A, const TensorPtr &B) {
    return elementwise_binary(A, B, [](float a, float b) { return a - b; });
  }

  TensorPtr tensor_mul(const TensorPtr &A, const TensorPtr &B) {
    return elementwise_binary(A, B, [](float a, float b) { return a * b; });
  }

  // 一元逐元素操作（示例：ReLU/GELU 需在后续扩展，这里先提供通用接口）
  TensorPtr elementwise_unary(const TensorPtr &A,
                              const std::function<float(float)> &op) {
    auto ctx = A->ctx.lock();
    if (!ctx) throw std::runtime_error("ctx expired");
    auto R = Tensor::create(ctx, A->shape, DType::FLOAT32);
    size_t n = A->nelements();
    for (size_t i = 0; i < n; ++i) {
      float a = A->get_as_float_flat(i);
      R->set_from_float_flat(i, op(a));
    }
    return R;
  }

  // Specialized Conv3d for Qwen-VL patch embedding
  // in_channels=3 -> out_channels=1280
  // kernel=(2,14,14), stride=(2,14,14), bias=False
  // Input X shape: {C=3, D, H, W}; Weight W: {Cout=1280, Cin=3, Kd=2, Kh=14, Kw=14}
  // Output: {Cout, D_out=D/2, H_out=H/14, W_out=W/14}
  static TensorPtr conv3d_k21414_s21414(const TensorPtr &X,
                                        const TensorPtr &W) {
    if (!X || !W)
      throw std::runtime_error("Conv3d: input or weight is null");
    auto ctx = X->ctx.lock();
    if (!ctx)
      throw std::runtime_error("Conv3d: context expired");
    if (X->shape.size() != 4)
      throw std::runtime_error("Conv3d: X must be 4D {C,D,H,W}");
    if (W->shape.size() != 5)
      throw std::runtime_error("Conv3d: W must be 5D {Cout,Cin,Kd,Kh,Kw}");
    const int Cin = X->shape[0];
    const int D = X->shape[1];
    const int H = X->shape[2];
    const int Ww = X->shape[3];
    const int Cout = W->shape[0];
    const int Win = W->shape[1];
    const int Kd = W->shape[2];
    const int Kh = W->shape[3];
    const int Kw = W->shape[4];
    if (Cin != 3 || Win != 3 || Cout != 1280 || Kd != 2 || Kh != 14 || Kw != 14)
      throw std::runtime_error(
          "Conv3d: shape mismatch, expected Cin=3,Cout=1280,K=(2,14,14)");
    if ((D % Kd) != 0 || (H % Kh) != 0 || (Ww % Kw) != 0)
      throw std::runtime_error(
          "Conv3d: input dims must be divisible by kernel dims");
    const int Dout = D / Kd;
    const int Hout = H / Kh;
    const int Wout = Ww / Kw;
    TensorPtr Y = Tensor::create(ctx, {Cout, Dout, Hout, Wout}, DType::FLOAT32);
    for (int oc = 0; oc < Cout; ++oc) {
      for (int od = 0; od < Dout; ++od) {
        const int d0 = od * Kd;
        for (int oh = 0; oh < Hout; ++oh) {
          const int h0 = oh * Kh;
          for (int ow = 0; ow < Wout; ++ow) {
            const int w0 = ow * Kw;
            double acc = 0.0;
            for (int c = 0; c < Cin; ++c) {
              for (int kd = 0; kd < Kd; ++kd) {
                for (int kh = 0; kh < Kh; ++kh) {
                  for (int kw = 0; kw < Kw; ++kw) {
                    size_t xi = (size_t)((((c * D) + (d0 + kd)) * H +
                                          (h0 + kh)) * Ww + (w0 + kw));
                    size_t wi = (size_t)(((((oc * Cin) + c) * Kd + kd) * Kh +
                                          kh) * Kw + kw);
                    float xv = X->get_as_float_flat(xi);
                    float wv = W->get_as_float_flat(wi);
                    acc += (double)xv * (double)wv;
                  }
                }
              }
            }
            size_t yi = (size_t)((((oc * Dout) + od) * Hout + oh) * Wout + ow);
            Y->set_from_float_flat(yi, (float)acc);
          }
        }
      }
    }
    return Y;
  }

  // Flatten Conv3d output {Cout, Dout, Hout, Wout} into tokens {Ntoks, Cout}
  static TensorPtr flatten_conv3d_tokens(const TensorPtr &Y) {
    if (!Y)
      throw std::runtime_error("flatten_conv3d_tokens: Y is null");
    auto ctx = Y->ctx.lock();
    if (!ctx)
      throw std::runtime_error("flatten_conv3d_tokens: context expired");
    if (Y->shape.size() != 4)
      throw std::runtime_error(
          "flatten_conv3d_tokens: Y must be 4D {Cout,Dout,Hout,Wout}");
    const int Cout = Y->shape[0];
    const int Dout = Y->shape[1];
    const int Hout = Y->shape[2];
    const int Wout = Y->shape[3];
    const int Ntoks = Dout * Hout * Wout;
    TensorPtr T = Tensor::create(ctx, {Ntoks, Cout}, DType::FLOAT32);
    int t = 0;
    for (int od = 0; od < Dout; ++od) {
      for (int oh = 0; oh < Hout; ++oh) {
        for (int ow = 0; ow < Wout; ++ow) {
          for (int oc = 0; oc < Cout; ++oc) {
            size_t yi = (size_t)((((oc * Dout) + od) * Hout + oh) * Wout + ow);
            float v = Y->get_as_float_flat(yi);
            size_t ti = (size_t)t * (size_t)Cout + (size_t)oc;
            T->set_from_float_flat(ti, v);
          }
          ++t;
        }
      }
    }
    return T;
  }

private:
  std::shared_ptr<Tensor> shared_from_this() {
    // 创建一个“空壳” shared_ptr：不增加引用计数，也不负责释放 this。
    // 第一个参数 this 作为裸指针传入，第二个参数是空删除器 lambda，
    // 因此当这个 shared_ptr 析构时不会 delete this，从而避免重复释放。
    // 典型用途：在对象内部把 this 包装成 shared_ptr 返回给调用者，
    // 让外部代码拿到一个安全的 shared_ptr，但生命周期仍由原本的 shared_ptr
    // 管理。
    return std::shared_ptr<Tensor>(this, [](Tensor *) {});
  }

  // simple matmul kernel
  void tensor_matmul_naive(const TensorPtr &A, const TensorPtr &B,
                           const TensorPtr &C) {
    assert(A && B && C);
    assert(A->dtype == DType::FLOAT32 && B->dtype == DType::FLOAT32 &&
           C->dtype == DType::FLOAT32);
    assert(A->shape.size() == 2 && B->shape.size() == 2 &&
           C->shape.size() == 2);

    int m = A->shape[0];
    int k = A->shape[1];
    int kb = B->shape[0];
    int n = B->shape[1];
    assert(k == kb);
    assert(C->shape[0] == m && C->shape[1] == n);

    const float *a = reinterpret_cast<const float *>(A->data);
    const float *b = reinterpret_cast<const float *>(B->data);
    float *c = reinterpret_cast<float *>(C->data);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int p = 0; p < k; ++p) {
          acc += a[i * k + p] * b[p * n + j];
        }
        c[i * n + j] = acc;
      }
    }
  }
};
} // namespace ow::nn