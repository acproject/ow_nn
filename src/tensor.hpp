#pragma once
#include "../include/common.h"
#include "context.hpp"
#include "thread_pool.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// 前缀声明，用于定义下面的智能指针
namespace ow::nn {
struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
struct QuantParams {
  float scale = 1.0f;
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

  static std::vector<int> calc_strides(const std::vector<int> &shape) {
    std::vector<int> s(shape.size());
    int acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      s[i] = acc;
      acc *= shape[i];
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
    if (new_ne != nelements())
      throw std::runtime_error("reshape size mismatch");
    if (!is_contiguous_row_major())
      throw std::runtime_error("reshape_view requires contiguous tensor");
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
    raw_view->data = static_cast<uint8_t *>(data) + offset_bytes;
    raw_view->ctx = ctx;
    // share same lifetime as self
    return TensorPtr(self, raw_view);
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
  float get_as_float_flat(size_t index) const {
    if (dtype == DType::FLOAT32) {
      const float *p = reinterpret_cast<const float *>(data);
      return p[index];
    }
    if (dtype == DType::FP16) {
      const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
      return fp16_to_float(p[index]);
    }
    if (dtype == DType::INT32) {
      const int32_t *p = reinterpret_cast<const int32_t *>(data);
      return float(p[index]);
    }
    if (dtype == DType::INT8) {
      const int8_t *p = reinterpret_cast<const int8_t *>(data);
      return float(p[index]);
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
  static TensorPtr matmul_blocked_mt(const TensorPtr &A, const TensorPtr &B,
                                     size_t block_m = 64, size_t block_n = 64,
                                     size_t block_k = 64, size_t nthreads = 0) {
    if (A->shape.size() != 2 || B->shape.size() != 2)
      throw std::runtime_error("matmul: need 2D");
    int m = A->shape[0];
    int k = A->shape[1];
    int kb = B->shape[0];
    int n = B->shape[1];

    if (k != kb)
      throw std::runtime_error("matmul dim");
    auto ctx = A->ctx.lock();
    if (!ctx)
      throw std::runtime_error("ctx expired");
    auto R = Tensor::create(ctx, {m, n}, DType::FLOAT32);
    // Extract raw pointers for float-able access via
    // get_as_float_flat/set_from_float_flat
    if (nthreads == 0)
      nthreads = std::max<size_t>(1, std::thread::hardware_concurrency());
    ThreadPool tp(nthreads);
    // partition i (rows) into tasks per block
    std::vector<std::future<void>> futs;
    for (int i0 = 0; i0 < m; i0 += (int)block_m) {
      int i1 = std::min(m, i0 + (int)block_m);
      futs.emplace_back(tp.submit([=, &A, &B, &R](void) -> void {
        for (int j0 = 0; j0 < n; j0 += (int)block_n) {
          int j1 = std::min(n, j0 + (int)block_n);
          for (int p0 = 0; p0 < k; p0 += (int)block_k) {
            int p1 = std::min(k, p0 + (int)block_k);
            // compute blcok C[i0:i1, j0:j1] += A[i0:i1, p0:p1] * B[p0:p1,
            // j0:j1]
            for (int ii = i0; ii < i1; ++ii) {
              for (int jj = j0; jj < j1; ++jj) {
                float acc = 0.0f;
                // if p0==0 and p1==k and ii==i0 and jj=j0 maybe can read
                // previous value; but accumulate
                for (int pp = p0; pp < p1; ++pp) {
                  size_t ai = (size_t)ii * k + pp;
                  size_t bi = (size_t)pp * n + jj;
                  float av = A->get_as_float_flat(ai);
                  float bv = B->get_as_float_flat(bi);
                  acc += av * bv;
                }

                // accumulate into R (read previous and add)
                size_t ri = (size_t)ii * n + jj;
                float prev = R->get_as_float_flat(ri);
                R->set_from_float_flat(ri, prev + acc);
              }
            }
          }
        }
      }));
    }
    // wait
    for (auto &f : futs)
      f.get();
    return R;
  }

  // ----------------------- elementwise ops (flattend)
  // ---------------------------
  TensorPtr elementwise_binary(const TensorPtr &A, const TensorPtr &B,
                               const std::function<float(float, float)> &op) {
    if (A->shape != B->shape)
      throw std::runtime_error("shape mismatch");
    auto ctx = A->ctx.lock();
    if (!ctx)
      throw std::runtime_error("ctx expired");
    auto R = Tensor::create(ctx, A->shape, DType::FLOAT32);
    size_t n = A->nelements();
    for (size_t i = 0; i < n; ++i) {
      float a = A->get_as_float_flat(i);
      float b = B->get_as_float_flat(i);
      R->set_from_float_flat(i, op(a, b));
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