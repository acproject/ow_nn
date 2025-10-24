#pragma once
#include "cgraph.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ow::nn {

// -------------------- helpers --------------------
static inline std::shared_ptr<Context> get_ctx_or_throw(const TensorPtr &T) {
  auto ctx = T->ctx.lock();
  if (!ctx)
    throw std::runtime_error("ctx expired");
  return ctx;
}

static inline int
attr_i(const std::unordered_map<std::string, std::string> &attrs,
       const std::string &key, int defv) {
  auto it = attrs.find(key);
  if (it == attrs.end())
    return defv;
  return std::stoi(it->second);
}
static inline float
attr_f(const std::unordered_map<std::string, std::string> &attrs,
       const std::string &key, float defv) {
  auto it = attrs.find(key);
  if (it == attrs.end())
    return defv;
  return std::stof(it->second);
}
static inline bool
attr_b(const std::unordered_map<std::string, std::string> &attrs,
       const std::string &key, bool defv) {
  auto it = attrs.find(key);
  if (it == attrs.end())
    return defv;
  const std::string &v = it->second;
  return (v == "1" || v == "true" || v == "True" || v == "YES" || v == "yes");
}

// -------------------- elementwise activations --------------------
inline TensorPtr relu(const TensorPtr &X) {
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t n = X->nelements();
  for (size_t i = 0; i < n; ++i) {
    float v = X->get_as_float_flat(i);
    Y->set_from_float_flat(i, v > 0.0f ? v : 0.0f);
  }
  return Y;
}

inline TensorPtr sigmoid(const TensorPtr &X) {
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t n = X->nelements();
  for (size_t i = 0; i < n; ++i) {
    float v = X->get_as_float_flat(i);
    float s = 1.0f / (1.0f + std::exp(-v));
    Y->set_from_float_flat(i, s);
  }
  return Y;
}

inline TensorPtr tanh_act(const TensorPtr &X) {
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t n = X->nelements();
  for (size_t i = 0; i < n; ++i) {
    float v = X->get_as_float_flat(i);
    Y->set_from_float_flat(i, std::tanh(v));
  }
  return Y;
}

inline TensorPtr gelu(const TensorPtr &X, bool approximate = true) {
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t n = X->nelements();
  if (approximate) {
    const float k0 = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < n; ++i) {
      float x = X->get_as_float_flat(i);
      float x3 = x * x * x;
      float t = std::tanh(k0 * (x + 0.044715f * x3));
      float y = 0.5f * x * (1.0f + t);
      Y->set_from_float_flat(i, y);
    }
  } else {
    // exact using erf
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < n; ++i) {
      float x = X->get_as_float_flat(i);
      float y = 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
      Y->set_from_float_flat(i, y);
    }
  }
  return Y;
}

inline TensorPtr silu(const TensorPtr &X) { // x * sigmoid(x)
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t n = X->nelements();
  for (size_t i = 0; i < n; ++i) {
    float x = X->get_as_float_flat(i);
    float s = 1.0f / (1.0f + std::exp(-x));
    Y->set_from_float_flat(i, x * s);
  }
  return Y;
}

inline TensorPtr softmax(const TensorPtr &X, int axis = -1) {
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("softmax requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  size_t r = X->shape.size();
  int ax = axis < 0 ? int(r) + axis : axis;
  if (ax < 0 || ax >= int(r))
    throw std::runtime_error("softmax axis out of range");
  int dim = X->shape[ax];
  size_t outer = 1;
  for (int i = 0; i < ax; ++i)
    outer *= (size_t)X->shape[i];
  size_t inner = 1;
  for (size_t i = ax + 1; i < r; ++i)
    inner *= (size_t)X->shape[i];
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t seg_len = (size_t)dim * inner;
  for (size_t o = 0; o < outer; ++o) {
    size_t base = o * seg_len;
    for (size_t in = 0; in < inner; ++in) {
      // max
      float m = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < dim; ++j) {
        size_t idx = base + (size_t)j * inner + in;
        m = std::max(m, X->get_as_float_flat(idx));
      }
      // sum exp
      float s = 0.0f;
      for (int j = 0; j < dim; ++j) {
        size_t idx = base + (size_t)j * inner + in;
        s += std::exp(X->get_as_float_flat(idx) - m);
      }
      float invs = s > 0 ? (1.0f / s) : 0.0f;
      for (int j = 0; j < dim; ++j) {
        size_t idx = base + (size_t)j * inner + in;
        float e = std::exp(X->get_as_float_flat(idx) - m) * invs;
        Y->set_from_float_flat(idx, e);
      }
    }
  }
  return Y;
}

// -------------------- ComputeGraph node factories (activations)
// --------------------
inline OpNode make_relu_node(const std::string &in, const std::string &out) {
  OpNode n;
  n.name = "relu";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return relu(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- BatchNorm --------------------
inline TensorPtr batch_norm(const TensorPtr &X, const TensorPtr &gamma,
                            const TensorPtr &beta,
                            const TensorPtr &running_mean,
                            const TensorPtr &running_var, float eps = 1e-5f) {
  if (X->shape.size() != 4 && X->shape.size() != 3)
    throw std::runtime_error("batch_norm expects rank 3 or 4 (CHW/NCHW)");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("batch_norm requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int N = 1, C, H, W;
  if (X->shape.size() == 4) {
    N = X->shape[0];
    C = X->shape[1];
    H = X->shape[2];
    W = X->shape[3];
  } else {
    C = X->shape[0];
    H = X->shape[1];
    W = X->shape[2];
  }
  if ((int)gamma->shape.size() != 1 || (int)beta->shape.size() != 1 ||
      (int)running_mean->shape.size() != 1 ||
      (int)running_var->shape.size() != 1)
    throw std::runtime_error("batch_norm: gamma/beta/mean/var must be 1D");
  if (gamma->shape[0] != C || beta->shape[0] != C ||
      running_mean->shape[0] != C || running_var->shape[0] != C)
    throw std::runtime_error("batch_norm: parameter size mismatch with C");
  std::vector<int> out_shape = X->shape;
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  if (X->shape.size() == 4) {
    size_t XinW = (size_t)H * (size_t)W;
    size_t XcW = (size_t)C * XinW;
    size_t YonW = (size_t)H * (size_t)W;
    size_t YcW = (size_t)C * YonW;
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        float g = gamma->get_as_float_flat((size_t)c);
        float b = beta->get_as_float_flat((size_t)c);
        float m = running_mean->get_as_float_flat((size_t)c);
        float v = running_var->get_as_float_flat((size_t)c);
        float invstd = 1.0f / std::sqrt(v + eps);
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            size_t idx = (size_t)n * XcW + (size_t)c * XinW +
                         (size_t)h * (size_t)W + (size_t)w;
            float x = X->get_as_float_flat(idx);
            float y = (x - m) * invstd * g + b;
            size_t oidx = (size_t)n * YcW + (size_t)c * YonW +
                          (size_t)h * (size_t)W + (size_t)w;
            Y->set_from_float_flat(oidx, y);
          }
        }
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      float g = gamma->get_as_float_flat((size_t)c);
      float b = beta->get_as_float_flat((size_t)c);
      float m = running_mean->get_as_float_flat((size_t)c);
      float v = running_var->get_as_float_flat((size_t)c);
      float invstd = 1.0f / std::sqrt(v + eps);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          size_t idx = (size_t)c * (size_t)H * (size_t)W +
                       (size_t)h * (size_t)W + (size_t)w;
          float x = X->get_as_float_flat(idx);
          float y = (x - m) * invstd * g + b;
          size_t oidx = (size_t)c * (size_t)H * (size_t)W +
                        (size_t)h * (size_t)W + (size_t)w;
          Y->set_from_float_flat(oidx, y);
        }
      }
    }
  }
  return Y;
}

inline OpNode
make_batch_norm_node(const std::string &in, const std::string &gamma,
                     const std::string &beta, const std::string &mean,
                     const std::string &var, const std::string &out) {
  OpNode n;
  n.name = "batch_norm";
  n.inputs = {in, gamma, beta, mean, var};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    float eps = attr_f(attrs, "eps", 1e-5f);
    return batch_norm(ins[0], ins[1], ins[2], ins[3], ins[4], eps);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    auto td = ids[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- LayerNorm --------------------
inline TensorPtr layer_norm(const TensorPtr &X, const TensorPtr &gamma,
                            const TensorPtr &beta, float eps = 1e-5f,
                            int axis = -1) {
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("layer_norm requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int r = (int)X->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r)
    throw std::runtime_error("layer_norm axis out of range");
  int dim = X->shape[ax];
  if ((int)gamma->shape.size() != 1 || (int)beta->shape.size() != 1 ||
      gamma->shape[0] != dim || beta->shape[0] != dim)
    throw std::runtime_error(
        "layer_norm: gamma/beta size mismatch with axis dim");
  size_t outer = 1;
  for (int i = 0; i < ax; ++i)
    outer *= (size_t)X->shape[i];
  size_t inner = 1;
  for (int i = ax + 1; i < r; ++i)
    inner *= (size_t)X->shape[i];
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t seg_len = (size_t)dim * inner;
  for (size_t o = 0; o < outer; ++o) {
    size_t base = o * seg_len;
    for (size_t in = 0; in < inner; ++in) {
      float mean = 0.0f;
      for (int j = 0; j < dim; ++j)
        mean += X->get_as_float_flat(base + (size_t)j * inner + in);
      mean /= (float)dim;
      float var = 0.0f;
      for (int j = 0; j < dim; ++j) {
        float v = X->get_as_float_flat(base + (size_t)j * inner + in) - mean;
        var += v * v;
      }
      var /= (float)dim;
      float invstd = 1.0f / std::sqrt(var + eps);
      for (int j = 0; j < dim; ++j) {
        float g = gamma->get_as_float_flat((size_t)j);
        float b = beta->get_as_float_flat((size_t)j);
        float x = X->get_as_float_flat(base + (size_t)j * inner + in);
        float y = (x - mean) * invstd * g + b;
        Y->set_from_float_flat(base + (size_t)j * inner + in, y);
      }
    }
  }
  return Y;
}

inline OpNode make_layer_norm_node(const std::string &in,
                                   const std::string &gamma,
                                   const std::string &beta,
                                   const std::string &out, int axis = -1) {
  OpNode n;
  n.name = "layer_norm";
  n.inputs = {in, gamma, beta};
  n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &ins,
                const std::unordered_map<std::string, std::string> &attrs) {
    float eps = attr_f(attrs, "eps", 1e-5f);
    int ax = axis;
    if (attrs.find("axis") != attrs.end())
      ax = attr_i(attrs, "axis", axis);
    return layer_norm(ins[0], ins[1], ins[2], eps, ax);
  };
  n.infer = [axis](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    auto td = ids[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- Dropout --------------------
inline TensorPtr dropout(const TensorPtr &X, float p, bool training = true,
                         unsigned int seed = 42) {
  if (p < 0.0f || p >= 1.0f)
    throw std::runtime_error("dropout: p in [0,1)");
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t n = X->nelements();
  if (!training) {
    float scale = 1.0f - p;
    for (size_t i = 0; i < n; ++i) {
      float v = X->get_as_float_flat(i);
      Y->set_from_float_flat(i, v * scale);
    }
    return Y;
  }
  std::mt19937 rng(seed);
  std::bernoulli_distribution keep(1.0f - p);
  float inv_keep = (1.0f - p) > 0 ? 1.0f / (1.0f - p) : 0.0f;
  for (size_t i = 0; i < n; ++i) {
    bool k = keep(rng);
    float v = X->get_as_float_flat(i);
    Y->set_from_float_flat(i, k ? (v * inv_keep) : 0.0f);
  }
  return Y;
}

inline OpNode make_dropout_node(const std::string &in, const std::string &out) {
  OpNode n;
  n.name = "dropout";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    float p = attr_f(attrs, "p", 0.0f);
    bool training = attr_b(attrs, "training", false);
    unsigned int seed = (unsigned int)attr_i(attrs, "seed", 42);
    return dropout(ins[0], p, training, seed);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    auto td = ids[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- Residual --------------------
inline TensorPtr residual_add(const TensorPtr &A, const TensorPtr &B,
                              float alpha = 1.0f) {
  // R = A + alpha * B
  auto ctx = get_ctx_or_throw(A);
  auto scaledB = B; // scale B elementwise
  if (alpha != 1.0f) {
    auto Y = Tensor::create(ctx, B->shape, DType::FLOAT32);
    size_t n = B->nelements();
    for (size_t i = 0; i < n; ++i) {
      Y->set_from_float_flat(i, B->get_as_float_flat(i) * alpha);
    }
    scaledB = Y;
  }
  // use existing broadcasting-compatible add
  return A->tensor_add(A, scaledB);
}

inline OpNode make_residual_node(const std::string &a, const std::string &b,
                                 const std::string &out) {
  OpNode n;
  n.name = "residual_add";
  n.inputs = {a, b};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    float alpha = attr_f(attrs, "alpha", 1.0f);
    return residual_add(ins[0], ins[1], alpha);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    // broadcast infer: use helper from Tensor if needed; for simplicity assume
    // same shape
    auto td = ids[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- Conv2d (NCHW/CHW) --------------------
inline TensorPtr conv2d(const TensorPtr &X, const TensorPtr &W,
                        const TensorPtr &bias, int sh = 1, int sw = 1,
                        int ph = 0, int pw = 0, int dh = 1, int dw = 1,
                        int groups = 1) {
  if (!X->is_contiguous_row_major() || !W->is_contiguous_row_major())
    throw std::runtime_error("conv2d requires contiguous inputs");
  if (groups != 1)
    throw std::runtime_error("conv2d: groups>1 not implemented");
  if ((int)W->shape.size() != 4)
    throw std::runtime_error("conv2d: W shape [Cout,Cin,Kh,Kw]");
  int N = 1, C, H, W_in;
  if (X->shape.size() == 4) {
    N = X->shape[0];
    C = X->shape[1];
    H = X->shape[2];
    W_in = X->shape[3];
  } else if (X->shape.size() == 3) {
    C = X->shape[0];
    H = X->shape[1];
    W_in = X->shape[2];
  } else
    throw std::runtime_error("conv2d expects rank 3 or 4");
  int Cout = W->shape[0];
  int Cin = W->shape[1];
  int Kh = W->shape[2];
  int Kw = W->shape[3];
  if (Cin != C)
    throw std::runtime_error("conv2d: Cin mismatch");
  if (bias && (int)bias->shape.size() != 1)
    throw std::runtime_error("conv2d: bias must be 1D");
  if (bias && bias->shape[0] != Cout)
    throw std::runtime_error("conv2d: bias size mismatch");
  int eff_kh = dh * (Kh - 1) + 1;
  int eff_kw = dw * (Kw - 1) + 1;
  int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
  int Wo = std::max(0, (W_in + 2 * pw - eff_kw) / sw + 1);
  std::vector<int> out_shape = (X->shape.size() == 4)
                                   ? std::vector<int>{N, Cout, Ho, Wo}
                                   : std::vector<int>{Cout, Ho, Wo};
  auto ctx = get_ctx_or_throw(X);
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  if (X->shape.size() == 4) {
    size_t XinW = (size_t)H * (size_t)W_in;
    size_t XcW = (size_t)C * XinW;
    size_t YonW = (size_t)Ho * (size_t)Wo;
    size_t YcW = (size_t)Cout * YonW;
    size_t WcW = (size_t)Cin * (size_t)Kh * (size_t)Kw;
    size_t WhW = (size_t)Kh * (size_t)Kw;
    for (int n = 0; n < N; ++n) {
      for (int oc = 0; oc < Cout; ++oc) {
        for (int oh = 0; oh < Ho; ++oh) {
          for (int ow = 0; ow < Wo; ++ow) {
            float acc = bias ? bias->get_as_float_flat((size_t)oc) : 0.0f;
            for (int ic = 0; ic < Cin; ++ic) {
              for (int kh_i = 0; kh_i < Kh; ++kh_i) {
                for (int kw_i = 0; kw_i < Kw; ++kw_i) {
                  int ih = oh * sh - ph + kh_i * dh;
                  int iw = ow * sw - pw + kw_i * dw;
                  if (ih >= 0 && ih < H && iw >= 0 && iw < W_in) {
                    size_t xidx = (size_t)n * XcW + (size_t)ic * XinW +
                                  (size_t)ih * (size_t)W_in + (size_t)iw;
                    size_t widx = (size_t)oc * WcW + (size_t)ic * WhW +
                                  (size_t)kh_i * (size_t)Kw + (size_t)kw_i;
                    acc +=
                        X->get_as_float_flat(xidx) * W->get_as_float_flat(widx);
                  }
                }
              }
            }
            size_t yidx = (size_t)n * YcW + (size_t)oc * YonW +
                          (size_t)oh * (size_t)Wo + (size_t)ow;
            Y->set_from_float_flat(yidx, acc);
          }
        }
      }
    }
  } else {
    size_t XinW = (size_t)H * (size_t)W_in;
    size_t WcW = (size_t)Cin * (size_t)Kh * (size_t)Kw;
    size_t WhW = (size_t)Kh * (size_t)Kw;
    for (int oc = 0; oc < Cout; ++oc) {
      for (int oh = 0; oh < Ho; ++oh) {
        for (int ow = 0; ow < Wo; ++ow) {
          float acc = bias ? bias->get_as_float_flat((size_t)oc) : 0.0f;
          for (int ic = 0; ic < Cin; ++ic) {
            for (int kh_i = 0; kh_i < Kh; ++kh_i) {
              for (int kw_i = 0; kw_i < Kw; ++kw_i) {
                int ih = oh * sh - ph + kh_i * dh;
                int iw = ow * sw - pw + kw_i * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W_in) {
                  size_t xidx = (size_t)ic * XinW + (size_t)ih * (size_t)W_in +
                                (size_t)iw;
                  size_t widx = (size_t)oc * WcW + (size_t)ic * WhW +
                                (size_t)kh_i * (size_t)Kw + (size_t)kw_i;
                  acc +=
                      X->get_as_float_flat(xidx) * W->get_as_float_flat(widx);
                }
              }
            }
          }
          size_t yidx = (size_t)oc * (size_t)Ho * (size_t)Wo +
                        (size_t)oh * (size_t)Wo + (size_t)ow;
          Y->set_from_float_flat(yidx, acc);
        }
      }
    }
  }
  return Y;
}

inline OpNode make_conv2d_node(const std::string &in, const std::string &weight,
                               const std::string &out,
                               const std::string &bias_optional = "") {
  OpNode n;
  n.name = "conv2d";
  n.inputs = bias_optional.empty()
                 ? std::vector<std::string>{in, weight}
                 : std::vector<std::string>{in, weight, bias_optional};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    int sh = attr_i(attrs, "stride_h", 1);
    int sw = attr_i(attrs, "stride_w", 1);
    int ph = attr_i(attrs, "pad_h", 0);
    int pw = attr_i(attrs, "pad_w", 0);
    int dh = attr_i(attrs, "dilation_h", 1);
    int dw = attr_i(attrs, "dilation_w", 1);
    int groups = attr_i(attrs, "groups", 1);
    const TensorPtr &X = ins[0];
    const TensorPtr &W = ins[1];
    const TensorPtr bias = (ins.size() >= 3) ? ins[2] : nullptr;
    return conv2d(X, W, bias, sh, sw, ph, pw, dh, dw, groups);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &attrs) {
    if (ids.size() < 2)
      throw std::runtime_error("conv2d infer: need X and W");
    auto Xd = ids[0];
    auto Wd = ids[1];
    if ((int)Wd.shape.size() != 4)
      throw std::runtime_error("conv2d infer: W rank 4");
    int Cout = Wd.shape[0];
    int Kh = Wd.shape[2];
    int Kw = Wd.shape[3];
    int sh = attr_i(attrs, "stride_h", 1);
    int sw = attr_i(attrs, "stride_w", 1);
    int ph = attr_i(attrs, "pad_h", 0);
    int pw = attr_i(attrs, "pad_w", 0);
    int dh = attr_i(attrs, "dilation_h", 1);
    int dw = attr_i(attrs, "dilation_w", 1);
    if (Xd.shape.size() == 4) {
      int H = Xd.shape[2], W = Xd.shape[3];
      int eff_kh = dh * (Kh - 1) + 1;
      int eff_kw = dw * (Kw - 1) + 1;
      int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
      int Wo = std::max(0, (W + 2 * pw - eff_kw) / sw + 1);
      Xd.shape = {Xd.shape[0], Cout, Ho, Wo};
      Xd.dtype = DType::FLOAT32;
      return Xd;
    } else if (Xd.shape.size() == 3) {
      int H = Xd.shape[1], W = Xd.shape[2];
      int eff_kh = dh * (Kh - 1) + 1;
      int eff_kw = dw * (Kw - 1) + 1;
      int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
      int Wo = std::max(0, (W + 2 * pw - eff_kw) / sw + 1);
      Xd.shape = {Cout, Ho, Wo};
      Xd.dtype = DType::FLOAT32;
      return Xd;
    } else {
      throw std::runtime_error("conv2d infer: X rank 3 or 4");
    }
  };
  return n;
}

inline OpNode make_sigmoid_node(const std::string &in, const std::string &out) {
  OpNode n;
  n.name = "sigmoid";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return sigmoid(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

inline OpNode make_tanh_node(const std::string &in, const std::string &out) {
  OpNode n;
  n.name = "tanh";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return tanh_act(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

inline OpNode make_gelu_node(const std::string &in, const std::string &out,
                             bool approximate = true) {
  OpNode n;
  n.name = approximate ? "gelu_approx" : "gelu_exact";
  n.inputs = {in};
  n.output = out;
  n.fn = [approximate](const std::vector<TensorPtr> &ins,
                       const std::unordered_map<std::string, std::string> &) {
    return gelu(ins[0], approximate);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

inline OpNode make_softmax_node(const std::string &in, const std::string &out,
                                int axis = -1) {
  OpNode n;
  n.name = "softmax";
  n.inputs = {in};
  n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &ins,
                const std::unordered_map<std::string, std::string> &) {
    return softmax(ins[0], axis);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0];
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- pooling --------------------
inline TensorPtr max_pool2d(const TensorPtr &X, int kh, int kw, int sh = 1,
                            int sw = 1, int ph = 0, int pw = 0, int dh = 1,
                            int dw = 1) {
  if (X->shape.size() != 4 && X->shape.size() != 3)
    throw std::runtime_error("max_pool2d expects rank 3 or 4 (CHW/NCHW)");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("max_pool2d requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int N = 1, C, H, W;
  if (X->shape.size() == 4) {
    N = X->shape[0];
    C = X->shape[1];
    H = X->shape[2];
    W = X->shape[3];
  } else {
    C = X->shape[0];
    H = X->shape[1];
    W = X->shape[2];
  }
  int eff_kh = dh * (kh - 1) + 1;
  int eff_kw = dw * (kw - 1) + 1;
  int Ho = (H + 2 * ph - eff_kh) / sh + 1;
  Ho = std::max(0, Ho);
  int Wo = (W + 2 * pw - eff_kw) / sw + 1;
  Wo = std::max(0, Wo);
  std::vector<int> out_shape = (X->shape.size() == 4)
                                   ? std::vector<int>{N, C, Ho, Wo}
                                   : std::vector<int>{C, Ho, Wo};
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  if (X->shape.size() == 4) {
    size_t XinW = (size_t)H * (size_t)W;
    size_t XcW = (size_t)C * XinW;
    size_t YonW = (size_t)Ho * (size_t)Wo;
    size_t YcW = (size_t)C * YonW;
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < Ho; ++oh) {
          for (int ow = 0; ow < Wo; ++ow) {
            float best = -std::numeric_limits<float>::infinity();
            for (int kh_i = 0; kh_i < kh; ++kh_i) {
              for (int kw_i = 0; kw_i < kw; ++kw_i) {
                int ih = oh * sh - ph + kh_i * dh;
                int iw = ow * sw - pw + kw_i * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  size_t idx = (size_t)n * XcW + (size_t)c * XinW +
                               (size_t)ih * (size_t)W + (size_t)iw;
                  best = std::max(best, X->get_as_float_flat(idx));
                }
              }
            }
            size_t oidx = (size_t)n * YcW + (size_t)c * YonW +
                          (size_t)oh * (size_t)Wo + (size_t)ow;
            Y->set_from_float_flat(oidx, best);
          }
        }
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < Ho; ++oh) {
        for (int ow = 0; ow < Wo; ++ow) {
          float best = -std::numeric_limits<float>::infinity();
          for (int kh_i = 0; kh_i < kh; ++kh_i) {
            for (int kw_i = 0; kw_i < kw; ++kw_i) {
              int ih = oh * sh - ph + kh_i * dh;
              int iw = ow * sw - pw + kw_i * dw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                size_t idx = (size_t)c * (size_t)H * (size_t)W +
                             (size_t)ih * (size_t)W + (size_t)iw;
                best = std::max(best, X->get_as_float_flat(idx));
              }
            }
          }
          size_t oidx = (size_t)c * (size_t)Ho * (size_t)Wo +
                        (size_t)oh * (size_t)Wo + (size_t)ow;
          Y->set_from_float_flat(oidx, best);
        }
      }
    }
  }
  return Y;
}

inline TensorPtr avg_pool2d(const TensorPtr &X, int kh, int kw, int sh = 1,
                            int sw = 1, int ph = 0, int pw = 0, int dh = 1,
                            int dw = 1) {
  if (X->shape.size() != 4 && X->shape.size() != 3)
    throw std::runtime_error("avg_pool2d expects rank 3 or 4 (CHW/NCHW)");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("avg_pool2d requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int N = 1, C, H, W;
  if (X->shape.size() == 4) {
    N = X->shape[0];
    C = X->shape[1];
    H = X->shape[2];
    W = X->shape[3];
  } else {
    C = X->shape[0];
    H = X->shape[1];
    W = X->shape[2];
  }
  int eff_kh = dh * (kh - 1) + 1;
  int eff_kw = dw * (kw - 1) + 1;
  int Ho = (H + 2 * ph - eff_kh) / sh + 1;
  Ho = std::max(0, Ho);
  int Wo = (W + 2 * pw - eff_kw) / sw + 1;
  Wo = std::max(0, Wo);
  std::vector<int> out_shape = (X->shape.size() == 4)
                                   ? std::vector<int>{N, C, Ho, Wo}
                                   : std::vector<int>{C, Ho, Wo};
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  if (X->shape.size() == 4) {
    size_t XinW = (size_t)H * (size_t)W;
    size_t XcW = (size_t)C * XinW;
    size_t YonW = (size_t)Ho * (size_t)Wo;
    size_t YcW = (size_t)C * YonW;
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < Ho; ++oh) {
          for (int ow = 0; ow < Wo; ++ow) {
            float sum = 0.0f;
            int count = 0;
            for (int kh_i = 0; kh_i < kh; ++kh_i) {
              for (int kw_i = 0; kw_i < kw; ++kw_i) {
                int ih = oh * sh - ph + kh_i * dh;
                int iw = ow * sw - pw + kw_i * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  size_t idx = (size_t)n * XcW + (size_t)c * XinW +
                               (size_t)ih * (size_t)W + (size_t)iw;
                  sum += X->get_as_float_flat(idx);
                  ++count;
                }
              }
            }
            float avg = (count > 0) ? (sum / (float)count) : 0.0f;
            size_t oidx = (size_t)n * YcW + (size_t)c * YonW +
                          (size_t)oh * (size_t)Wo + (size_t)ow;
            Y->set_from_float_flat(oidx, avg);
          }
        }
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < Ho; ++oh) {
        for (int ow = 0; ow < Wo; ++ow) {
          float sum = 0.0f;
          int count = 0;
          for (int kh_i = 0; kh_i < kh; ++kh_i) {
            for (int kw_i = 0; kw_i < kw; ++kw_i) {
              int ih = oh * sh - ph + kh_i * dh;
              int iw = ow * sw - pw + kw_i * dw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                size_t idx = (size_t)c * (size_t)H * (size_t)W +
                             (size_t)ih * (size_t)W + (size_t)iw;
                sum += X->get_as_float_flat(idx);
                ++count;
              }
            }
          }
          float avg = (count > 0) ? (sum / (float)count) : 0.0f;
          size_t oidx = (size_t)c * (size_t)Ho * (size_t)Wo +
                        (size_t)oh * (size_t)Wo + (size_t)ow;
          Y->set_from_float_flat(oidx, avg);
        }
      }
    }
  }
  return Y;
}

inline OpNode make_max_pool2d_node(const std::string &in,
                                   const std::string &out) {
  OpNode n;
  n.name = "max_pool2d";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    int kh = attr_i(attrs, "kernel_h", 2);
    int kw = attr_i(attrs, "kernel_w", 2);
    int sh = attr_i(attrs, "stride_h", 2);
    int sw = attr_i(attrs, "stride_w", 2);
    int ph = attr_i(attrs, "pad_h", 0);
    int pw = attr_i(attrs, "pad_w", 0);
    int dh = attr_i(attrs, "dilation_h", 1);
    int dw = attr_i(attrs, "dilation_w", 1);
    return max_pool2d(ins[0], kh, kw, sh, sw, ph, pw, dh, dw);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &attrs) {
    auto td = in_descs[0];
    int kh = attr_i(attrs, "kernel_h", 2);
    int kw = attr_i(attrs, "kernel_w", 2);
    int sh = attr_i(attrs, "stride_h", 2);
    int sw = attr_i(attrs, "stride_w", 2);
    int ph = attr_i(attrs, "pad_h", 0);
    int pw = attr_i(attrs, "pad_w", 0);
    int dh = attr_i(attrs, "dilation_h", 1);
    int dw = attr_i(attrs, "dilation_w", 1);
    int H_idx = (td.shape.size() == 4) ? 2 : 1;
    int W_idx = (td.shape.size() == 4) ? 3 : 2;
    int H = td.shape[H_idx], W = td.shape[W_idx];
    int eff_kh = dh * (kh - 1) + 1;
    int eff_kw = dw * (kw - 1) + 1;
    int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
    int Wo = std::max(0, (W + 2 * pw - eff_kw) / sw + 1);
    td.shape[H_idx] = Ho;
    td.shape[W_idx] = Wo;
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

inline OpNode make_avg_pool2d_node(const std::string &in,
                                   const std::string &out) {
  OpNode n;
  n.name = "avg_pool2d";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    int kh = attr_i(attrs, "kernel_h", 2);
    int kw = attr_i(attrs, "kernel_w", 2);
    int sh = attr_i(attrs, "stride_h", 2);
    int sw = attr_i(attrs, "stride_w", 2);
    int ph = attr_i(attrs, "pad_h", 0);
    int pw = attr_i(attrs, "pad_w", 0);
    int dh = attr_i(attrs, "dilation_h", 1);
    int dw = attr_i(attrs, "dilation_w", 1);
    return avg_pool2d(ins[0], kh, kw, sh, sw, ph, pw, dh, dw);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &attrs) {
    auto td = in_descs[0];
    int kh = attr_i(attrs, "kernel_h", 2);
    int kw = attr_i(attrs, "kernel_w", 2);
    int sh = attr_i(attrs, "stride_h", 2);
    int sw = attr_i(attrs, "stride_w", 2);
    int ph = attr_i(attrs, "pad_h", 0);
    int pw = attr_i(attrs, "pad_w", 0);
    int dh = attr_i(attrs, "dilation_h", 1);
    int dw = attr_i(attrs, "dilation_w", 1);
    int H_idx = (td.shape.size() == 4) ? 2 : 1;
    int W_idx = (td.shape.size() == 4) ? 3 : 2;
    int H = td.shape[H_idx], W = td.shape[W_idx];
    int eff_kh = dh * (kh - 1) + 1;
    int eff_kw = dw * (kw - 1) + 1;
    int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
    int Wo = std::max(0, (W + 2 * pw - eff_kw) / sw + 1);
    td.shape[H_idx] = Ho;
    td.shape[W_idx] = Wo;
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- flatten --------------------
inline TensorPtr flatten(const TensorPtr &X, int start_axis = 0,
                         int end_axis = -1) {
  int r = (int)X->shape.size();
  int sa = start_axis < 0 ? r + start_axis : start_axis;
  int ea = end_axis < 0 ? r + end_axis : end_axis;
  if (sa < 0 || ea >= r || sa > ea)
    throw std::runtime_error("flatten: axis out of range");
  std::vector<int> ns;
  ns.reserve(r - (ea - sa));
  for (int i = 0; i < sa; ++i)
    ns.push_back(X->shape[i]);
  int merged = 1;
  for (int i = sa; i <= ea; ++i)
    merged *= X->shape[i];
  ns.push_back(merged);
  for (int i = ea + 1; i < r; ++i)
    ns.push_back(X->shape[i]);
  if (X->is_contiguous_row_major())
    return X->reshape_view(ns);
  auto Xc = X->copy();
  return Xc->reshape_view(ns);
}

inline OpNode make_flatten_node(const std::string &in, const std::string &out) {
  OpNode n;
  n.name = "flatten";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    int sa = attr_i(attrs, "start_axis", 1);
    int ea = attr_i(attrs, "end_axis", -1);
    return flatten(ins[0], sa, ea);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &attrs) {
    auto td = in_descs[0];
    int r = (int)td.shape.size();
    int sa = attr_i(attrs, "start_axis", 1);
    int ea_def = -1; // store raw then normalize
    int ea = ea_def;
    if (attrs.find("end_axis") != attrs.end())
      ea = attr_i(attrs, "end_axis", -1);
    int sa_n = sa < 0 ? r + sa : sa;
    int ea_n = ea < 0 ? r + ea : ea;
    if (sa_n < 0 || ea_n >= r || sa_n > ea_n)
      throw std::runtime_error("flatten infer: axis out of range");
    std::vector<int> ns;
    ns.reserve(r - (ea_n - sa_n));
    for (int i = 0; i < sa_n; ++i)
      ns.push_back(td.shape[i]);
    int merged = 1;
    for (int i = sa_n; i <= ea_n; ++i)
      merged *= td.shape[i];
    ns.push_back(merged);
    for (int i = ea_n + 1; i < r; ++i)
      ns.push_back(td.shape[i]);
    td.shape = ns;
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- concat --------------------
inline TensorPtr concat(const std::vector<TensorPtr> &inputs, int axis) {
  if (inputs.empty())
    throw std::runtime_error("concat: empty inputs");
  int r = (int)inputs[0]->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r)
    throw std::runtime_error("concat: axis out of range");
  for (auto &t : inputs) {
    if ((int)t->shape.size() != r)
      throw std::runtime_error("concat: rank mismatch");
    if (!t->is_contiguous_row_major())
      throw std::runtime_error("concat requires contiguous inputs");
  }
  std::vector<int> out_shape = inputs[0]->shape;
  int sum_ax = 0;
  for (auto &t : inputs) {
    for (int i = 0; i < r; ++i) {
      if (i == ax)
        continue;
      if (t->shape[i] != out_shape[i])
        throw std::runtime_error("concat: non-axis dims mismatch");
    }
    sum_ax += t->shape[ax];
  }
  out_shape[ax] = sum_ax;
  auto ctx = get_ctx_or_throw(inputs[0]);
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  size_t outer = 1;
  for (int i = 0; i < ax; ++i)
    outer *= (size_t)out_shape[i];
  size_t inner = 1;
  for (int i = ax + 1; i < r; ++i)
    inner *= (size_t)out_shape[i];
  for (size_t o = 0; o < outer; ++o) {
    size_t out_base = o * (size_t)out_shape[ax] * inner;
    size_t run = 0;
    for (auto &t : inputs) {
      size_t dimk = (size_t)t->shape[ax];
      size_t in_base = o * dimk * inner;
      for (size_t j = 0; j < dimk; ++j) {
        for (size_t in = 0; in < inner; ++in) {
          size_t src = in_base + j * inner + in;
          size_t dst = out_base + (run + j) * inner + in;
          float v = t->get_as_float_flat(src);
          Y->set_from_float_flat(dst, v);
        }
      }
      run += dimk;
    }
  }
  return Y;
}

inline TensorPtr slice(const TensorPtr &X, int axis, int start, int length,
                       int step = 1) {
  if (!X)
    throw std::runtime_error("slice: null input");
  auto ctx = get_ctx_or_throw(X);
  int r = (int)X->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r)
    throw std::runtime_error("slice: axis out of range");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("slice requires contiguous input");
  int size_d = X->shape[ax];
  int s = start;
  if (s < 0)
    s += size_d;
  if (s < 0 || s >= size_d)
    throw std::runtime_error("slice: start out of range");
  int len = (length < 0) ? (size_d - s) : length;
  if (len <= 0 || s + len > size_d)
    throw std::runtime_error("slice: length out of range");
  if (step <= 0)
    throw std::runtime_error("slice: step must be positive");
  int out_len = (step == 1) ? len : (len + step - 1) / step;
  std::vector<int> out_shape = X->shape;
  out_shape[ax] = out_len;
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  size_t outer = 1;
  for (int i = 0; i < ax; ++i)
    outer *= (size_t)X->shape[i];
  size_t inner = 1;
  for (int i = ax + 1; i < r; ++i)
    inner *= (size_t)X->shape[i];
  for (size_t o = 0; o < outer; ++o) {
    size_t in_base = o * (size_t)size_d * inner;
    size_t out_base = o * (size_t)out_len * inner;
    for (int j = 0; j < out_len; ++j) {
      int src_j = s + j * step;
      if (src_j >= s + len)
        break;
      for (size_t in = 0; in < inner; ++in) {
        size_t src = in_base + (size_t)src_j * inner + in;
        size_t dst = out_base + (size_t)j * inner + in;
        float v = X->get_as_float_flat(src);
        Y->set_from_float_flat(dst, v);
      }
    }
  }
  return Y;
}

// View variant: returns a non-copying view with strided slicing
inline TensorPtr slice_view(const TensorPtr &X, int axis, int start, int length,
                            int step = 1) {
  if (!X)
    throw std::runtime_error("slice_view: null input");
  int r = (int)X->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r)
    throw std::runtime_error("slice_view: axis out of range");
  return X->slice_view_step(ax, start, length, step);
}

inline OpNode make_slice_node(const std::string &in, const std::string &out,
                              int axis, int start, int length, int step = 1) {
  OpNode n;
  n.name = "slice";
  n.inputs = {in};
  n.output = out;
  n.fn = [axis, start, length,
          step](const std::vector<TensorPtr> &vs,
                const std::unordered_map<std::string, std::string> &) {
    return slice(vs[0], axis, start, length, step);
  };
  n.infer = [axis, start, length,
             step](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("slice infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int ax = axis < 0 ? r + axis : axis;
    if (ax < 0 || ax >= r)
      throw std::runtime_error("slice infer: axis out of range");
    TensorDesc td = ids[0];
    int size_d = td.shape[ax];
    int s = start;
    if (s < 0)
      s += size_d;
    if (s < 0 || s >= size_d)
      throw std::runtime_error("slice infer: start out of range");
    int len = (length < 0) ? (size_d - s) : length;
    if (len <= 0 || s + len > size_d)
      throw std::runtime_error("slice infer: length out of range");
    int st = std::max(step, 1);
    int out_len = (st == 1) ? len : (len + st - 1) / st;
    td.shape[ax] = out_len;
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

inline OpNode make_slice_view_node(const std::string &in,
                                   const std::string &out, int axis, int start,
                                   int length, int step = 1) {
  OpNode n;
  n.name = "slice_view";
  n.inputs = {in};
  n.output = out;
  n.fn = [axis, start, length,
          step](const std::vector<TensorPtr> &vs,
                const std::unordered_map<std::string, std::string> &) {
    return slice_view(vs[0], axis, start, length, step);
  };
  n.infer = [axis, start, length,
             step](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("slice_view infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int ax = axis < 0 ? r + axis : axis;
    if (ax < 0 || ax >= r)
      throw std::runtime_error("slice_view infer: axis out of range");
    TensorDesc td = ids[0];
    int size_d = td.shape[ax];
    int s = start;
    if (s < 0)
      s += size_d;
    if (s < 0 || s >= size_d)
      throw std::runtime_error("slice_view infer: start out of range");
    int len = (length < 0) ? (size_d - s) : length;
    if (len <= 0 || s + len > size_d)
      throw std::runtime_error("slice_view infer: length out of range");
    int st = std::max(step, 1);
    int out_len = (st == 1) ? len : (len + st - 1) / st;
    td.shape[ax] = out_len;
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// ---------------- Shape utilities as ops ----------------
inline TensorPtr shape_of(const TensorPtr &X) {
  if (!X)
    throw std::runtime_error("shape_of: null input");
  auto ctx = get_ctx_or_throw(X);
  int r = (int)X->shape.size();
  auto S = Tensor::create(ctx, std::vector<int>{r}, DType::INT32);
  for (int i = 0; i < r; ++i) {
    S->set_from_float_flat(i, (float)X->shape[i]);
  }
  return S;
}

inline TensorPtr shape_slice(const TensorPtr &X, int start, int length,
                             int step = 1) {
  auto S = shape_of(X);
  return slice(S, 0, start, length, step);
}

inline OpNode make_shape_of_node(const std::string &in,
                                 const std::string &out) {
  OpNode n;
  n.name = "shape_of";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &vs,
            const std::unordered_map<std::string, std::string> &) {
    return shape_of(vs[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_of infer: needs 1 input");
    TensorDesc td;
    td.shape = {(int)ids[0].shape.size()};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return n;
}

inline OpNode make_shape_slice_node(const std::string &in,
                                    const std::string &out, int start,
                                    int length, int step = 1) {
  OpNode n;
  n.name = "shape_slice";
  n.inputs = {in};
  n.output = out;
  n.fn = [start, length, step](const std::vector<TensorPtr> &vs,
                               const std::unordered_map<std::string, std::string> &) {
    return shape_slice(vs[0], start, length, step);
  };
  n.infer = [start, length, step](const std::vector<TensorDesc> &ids,
                                  const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_slice infer: needs 1 input");
    int size_d = (int)ids[0].shape.size();
    int s = start;
    if (s < 0)
      s += size_d;
    if (s < 0 || s >= size_d)
      throw std::runtime_error("shape_slice infer: start out of range");
    int len = (length < 0) ? (size_d - s) : length;
    if (len <= 0 || s + len > size_d)
      throw std::runtime_error("shape_slice infer: length out of range");
    int st = std::max(step, 1);
    int out_len = (st == 1) ? len : (len + st - 1) / st;
    TensorDesc td;
    td.shape = {out_len};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return n;
}

// Convenience shape ops wrapping shape_slice
inline TensorPtr shape_all_but_last(const TensorPtr &X) {
  return shape_slice(X, 0, -1, 1);
}
inline TensorPtr shape_all_but_first(const TensorPtr &X) {
  return shape_slice(X, 1, -1, 1);
}
inline TensorPtr shape_first_n(const TensorPtr &X, int n) {
  return shape_slice(X, 0, n, 1);
}
inline TensorPtr shape_last_n(const TensorPtr &X, int n) {
  return shape_slice(X, -n, -1, 1);
}
inline TensorPtr shape_select(const TensorPtr &X, int index) {
  return shape_slice(X, index, 1, 1);
}
inline TensorPtr shape_rank(const TensorPtr &X) {
  auto ctx = get_ctx_or_throw(X);
  auto R = Tensor::create(ctx, std::vector<int>{1}, DType::INT32);
  R->set_from_float_flat(0, (float)X->shape.size());
  return R;
}

inline OpNode make_shape_all_but_last_node(const std::string &in,
                                           const std::string &out) {
  OpNode n;
  n.name = "shape_all_but_last";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &vs,
            const std::unordered_map<std::string, std::string> &) {
    return shape_all_but_last(vs[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_all_but_last infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int out_len = std::max(0, r - 1);
    TensorDesc td;
    td.shape = {out_len};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return n;
}

inline OpNode make_shape_all_but_first_node(const std::string &in,
                                            const std::string &out) {
  OpNode n;
  n.name = "shape_all_but_first";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &vs,
            const std::unordered_map<std::string, std::string> &) {
    return shape_all_but_first(vs[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_all_but_first infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int out_len = std::max(0, r - 1);
    TensorDesc td;
    td.shape = {out_len};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return n;
}

inline OpNode make_shape_first_n_node(const std::string &in,
                                      const std::string &out, int n) {
  OpNode nnode;
  nnode.name = "shape_first_n";
  nnode.inputs = {in};
  nnode.output = out;
  nnode.fn = [n](const std::vector<TensorPtr> &vs,
                 const std::unordered_map<std::string, std::string> &) {
    return shape_first_n(vs[0], n);
  };
  nnode.infer = [n](const std::vector<TensorDesc> &ids,
                    const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_first_n infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int out_len = std::max(0, std::min(n, r));
    TensorDesc td;
    td.shape = {out_len};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return nnode;
}

inline OpNode make_shape_last_n_node(const std::string &in,
                                     const std::string &out, int n) {
  OpNode nnode;
  nnode.name = "shape_last_n";
  nnode.inputs = {in};
  nnode.output = out;
  nnode.fn = [n](const std::vector<TensorPtr> &vs,
                 const std::unordered_map<std::string, std::string> &) {
    return shape_last_n(vs[0], n);
  };
  nnode.infer = [n](const std::vector<TensorDesc> &ids,
                    const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_last_n infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int out_len = std::max(0, std::min(n, r));
    TensorDesc td;
    td.shape = {out_len};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return nnode;
}

inline OpNode make_shape_select_node(const std::string &in,
                                     const std::string &out, int index) {
  OpNode n;
  n.name = "shape_select";
  n.inputs = {in};
  n.output = out;
  n.fn = [index](const std::vector<TensorPtr> &vs,
                 const std::unordered_map<std::string, std::string> &) {
    return shape_select(vs[0], index);
  };
  n.infer = [index](const std::vector<TensorDesc> &ids,
                    const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_select infer: needs 1 input");
    int r = (int)ids[0].shape.size();
    int ix = index < 0 ? r + index : index;
    if (ix < 0 || ix >= r)
      throw std::runtime_error("shape_select infer: index out of range");
    TensorDesc td;
    td.shape = {1};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return n;
}

inline OpNode make_shape_rank_node(const std::string &in,
                                   const std::string &out) {
  OpNode n;
  n.name = "shape_rank";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &vs,
            const std::unordered_map<std::string, std::string> &) {
    return shape_rank(vs[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("shape_rank infer: needs 1 input");
    TensorDesc td;
    td.shape = {1};
    td.strides = Tensor::calc_strides(td.shape);
    td.dtype = DType::INT32;
    return td;
  };
  return n;
}

inline OpNode make_concat_node(const std::vector<std::string> &ins,
                               const std::string &out, int axis) {
  OpNode n;
  n.name = "concat";
  n.inputs = ins;
  n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &vs,
                const std::unordered_map<std::string, std::string> &) {
    return concat(vs, axis);
  };
  n.infer = [axis](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    if (ids.empty())
      throw std::runtime_error("concat infer: empty inputs");
    int r = (int)ids[0].shape.size();
    int ax = axis < 0 ? r + axis : axis;
    if (ax < 0 || ax >= r)
      throw std::runtime_error("concat infer: axis out of range");
    TensorDesc td = ids[0];
    int sum_ax = 0;
    for (auto &d : ids) {
      if ((int)d.shape.size() != r)
        throw std::runtime_error("concat infer: rank mismatch");
      for (int i = 0; i < r; ++i) {
        if (i == ax)
          continue;
        if (d.shape[i] != td.shape[i])
          throw std::runtime_error("concat infer: non-axis dims mismatch");
      }
      sum_ax += d.shape[ax];
    }
    td.shape[ax] = sum_ax;
    td.dtype = DType::FLOAT32;
    return td;
  };
  return n;
}

// -------------------- Shape ops: unsqueeze / squeeze / permute
// --------------------
inline TensorPtr unsqueeze(const TensorPtr &X, int axis) {
  int r = (int)X->shape.size();
  int ax = axis < 0 ? r + 1 + axis : axis;
  if (ax < 0 || ax > r)
    throw std::runtime_error("unsqueeze: axis out of range");
  std::vector<int> ns = X->shape;
  ns.insert(ns.begin() + ax, 1);
  if (X->is_contiguous_row_major())
    return X->reshape_view(ns);
  return X->copy()->reshape_view(ns);
}

inline TensorPtr squeeze_axis(const TensorPtr &X, int axis) {
  int r = (int)X->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r)
    throw std::runtime_error("squeeze: axis out of range");
  if (X->shape[ax] != 1)
    throw std::runtime_error("squeeze: dimension to squeeze must be 1");
  std::vector<int> ns;
  ns.reserve(r - 1);
  for (int i = 0; i < r; ++i)
    if (i != ax)
      ns.push_back(X->shape[i]);
  if (ns.empty())
    ns.push_back(1);
  if (X->is_contiguous_row_major())
    return X->reshape_view(ns);
  return X->copy()->reshape_view(ns);
}

inline TensorPtr squeeze_all(const TensorPtr &X) {
  int r = (int)X->shape.size();
  std::vector<int> ns;
  ns.reserve(r);
  for (int i = 0; i < r; ++i)
    if (X->shape[i] != 1)
      ns.push_back(X->shape[i]);
  if (ns.empty())
    ns.push_back(1);
  if (X->is_contiguous_row_major())
    return X->reshape_view(ns);
  return X->copy()->reshape_view(ns);
}

inline TensorPtr permute_view(const TensorPtr &X,
                              const std::vector<int> &dims) {
  int r = (int)X->shape.size();
  if ((int)dims.size() != r)
    throw std::runtime_error("permute: dims size must equal rank");
  std::vector<int> new_shape(r), new_strides(r);
  std::vector<char> used(r, 0);
  for (int i = 0; i < r; ++i) {
    int d = dims[i];
    if (d < 0)
      d += r;
    if (d < 0 || d >= r)
      throw std::runtime_error("permute: dim index out of range");
    if (used[d])
      throw std::runtime_error("permute: repeated dim index");
    used[d] = 1;
    new_shape[i] = X->shape[d];
    new_strides[i] = X->strides[d];
  }
  return X->view(new_shape, new_strides, 0);
}

inline OpNode make_unsqueeze_node(const std::string &in, const std::string &out,
                                  int axis) {
  OpNode n;
  n.name = "unsqueeze";
  n.inputs = {in};
  n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &vs,
                const std::unordered_map<std::string, std::string> &) {
    return unsqueeze(vs[0], axis);
  };
  n.infer = [axis](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("unsqueeze infer: need 1 input");
    int r = (int)ids[0].shape.size();
    int ax = axis < 0 ? r + 1 + axis : axis;
    if (ax < 0 || ax > r)
      throw std::runtime_error("unsqueeze infer: axis out of range");
    auto ns = ids[0].shape;
    ns.insert(ns.begin() + ax, 1);
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = ns;
    td.strides = Tensor::calc_strides(ns);
    return td;
  };
  return n;
}

inline OpNode make_squeeze_node(const std::string &in, const std::string &out,
                                bool has_axis, int axis_if_any) {
  OpNode n;
  n.name = "squeeze";
  n.inputs = {in};
  n.output = out;
  n.fn = [has_axis,
          axis_if_any](const std::vector<TensorPtr> &vs,
                       const std::unordered_map<std::string, std::string> &) {
    if (has_axis)
      return squeeze_axis(vs[0], axis_if_any);
    return squeeze_all(vs[0]);
  };
  n.infer = [has_axis, axis_if_any](
                const std::vector<TensorDesc> &ids,
                const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("squeeze infer: need 1 input");
    auto ns = ids[0].shape;
    if (has_axis) {
      int r = (int)ns.size();
      int ax = axis_if_any < 0 ? r + axis_if_any : axis_if_any;
      if (ax < 0 || ax >= r)
        throw std::runtime_error("squeeze infer: axis out of range");
      if (ns[ax] != 1)
        throw std::runtime_error("squeeze infer: dim to squeeze must be 1");
      ns.erase(ns.begin() + ax);
      if (ns.empty())
        ns.push_back(1);
    } else {
      std::vector<int> tmp;
      tmp.reserve(ns.size());
      for (int d : ns)
        if (d != 1)
          tmp.push_back(d);
      ns.swap(tmp);
      if (ns.empty())
        ns.push_back(1);
    }
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = ns;
    td.strides = Tensor::calc_strides(ns);
    return td;
  };
  return n;
}

inline OpNode make_permute_node(const std::string &in, const std::string &out,
                                const std::vector<int> &dims) {
  OpNode n;
  n.name = "permute";
  n.inputs = {in};
  n.output = out;
  n.fn = [dims](const std::vector<TensorPtr> &vs,
                const std::unordered_map<std::string, std::string> &) {
    return permute_view(vs[0], dims);
  };
  n.infer = [dims](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    if (ids.size() != 1)
      throw std::runtime_error("permute infer: need 1 input");
    int r = (int)ids[0].shape.size();
    if ((int)dims.size() != r)
      throw std::runtime_error("permute infer: dims size != rank");
    std::vector<int> ns(r);
    std::vector<char> used(r, 0);
    for (int i = 0; i < r; ++i) {
      int d = dims[i];
      if (d < 0)
        d += r;
      if (d < 0 || d >= r)
        throw std::runtime_error("permute infer: dim out of range");
      if (used[d])
        throw std::runtime_error("permute infer: repeated dim");
      used[d] = 1;
      ns[i] = ids[0].shape[d];
    }
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = ns;
    td.strides = Tensor::calc_strides(ns);
    return td;
  };
  return n;
}

// -------------------- Linear / Matmul / Elementwise node factories
// --------------------
inline OpNode make_linear_node(const std::string &in, const std::string &weight,
                               const std::string &out,
                               const std::string &bias_optional = "") {
  OpNode n;
  n.name = "linear";
  n.inputs = bias_optional.empty()
                 ? std::vector<std::string>{in, weight}
                 : std::vector<std::string>{in, weight, bias_optional};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    const TensorPtr &X = ins[0];
    const TensorPtr &W = ins[1];
    const TensorPtr bias = (ins.size() >= 3) ? ins[2] : nullptr;
    if (bias)
      return Tensor::linear(X, W, bias);
    return Tensor::linear(X, W);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 2)
      throw std::runtime_error("linear infer: need X and W");
    auto Xd = ids[0];
    auto Wd = ids[1];
    if ((int)Xd.shape.size() != 2 || (int)Wd.shape.size() != 2)
      throw std::runtime_error("linear infer: X/W rank 2");
    int m = Xd.shape[0];
    int kx = Xd.shape[1];
    int kw = Wd.shape[0];
    int nw = Wd.shape[1];
    int n;
    // 支持 W 为 [k,n] 或 [n,k]（与 PyTorch 权重相兼容）
    if (kw == kx) {
      n = nw; // W: [k,n]
    } else if (nw == kx) {
      n = kw; // W: [n,k]
    } else {
      throw std::runtime_error("linear infer: K mismatch between X and W");
    }
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = {m, n};
    td.strides = Tensor::calc_strides(td.shape);
    return td;
  };
  return n;
}

inline OpNode make_matmul_node(const std::string &a, const std::string &b,
                               const std::string &out) {
  OpNode n;
  n.name = "matmul";
  n.inputs = {a, b};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return Tensor::matmul_cache_friendly(ins[0], ins[1]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 2)
      throw std::runtime_error("matmul infer: need A and B");
    auto Ad = ids[0];
    auto Bd = ids[1];
    if ((int)Ad.shape.size() != 2 || (int)Bd.shape.size() != 2)
      throw std::runtime_error("matmul infer: A/B rank 2");
    int m = Ad.shape[0];
    int kA = Ad.shape[1];
    int kB = Bd.shape[0];
    int n = Bd.shape[1];
    if (kA != kB)
      throw std::runtime_error("matmul infer: K mismatch");
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = {m, n};
    td.strides = Tensor::calc_strides(td.shape);
    return td;
  };
  return n;
}

inline OpNode make_add_node(const std::string &a, const std::string &b,
                            const std::string &out) {
  OpNode n;
  n.name = "add";
  n.inputs = {a, b};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return Tensor::tensor_add(ins[0], ins[1]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 2)
      throw std::runtime_error("add infer: need A and B");
    auto s = Tensor::broadcast_shape(ids[0].shape, ids[1].shape);
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = s;
    td.strides = Tensor::calc_strides(s);
    return td;
  };
  return n;
}

inline OpNode make_sub_node(const std::string &a, const std::string &b,
                            const std::string &out) {
  OpNode n;
  n.name = "sub";
  n.inputs = {a, b};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return Tensor::tensor_sub(ins[0], ins[1]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 2)
      throw std::runtime_error("sub infer: need A and B");
    auto s = Tensor::broadcast_shape(ids[0].shape, ids[1].shape);
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = s;
    td.strides = Tensor::calc_strides(s);
    return td;
  };
  return n;
}

inline OpNode make_mul_node(const std::string &a, const std::string &b,
                            const std::string &out) {
  OpNode n;
  n.name = "mul";
  n.inputs = {a, b};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return Tensor::tensor_mul(ins[0], ins[1]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 2)
      throw std::runtime_error("mul infer: need A and B");
    auto s = Tensor::broadcast_shape(ids[0].shape, ids[1].shape);
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = s;
    td.strides = Tensor::calc_strides(s);
    return td;
  };
  return n;
}

inline OpNode make_expand_node(const std::string &in, const std::string &out,
                               const std::vector<int> &new_shape) {
  OpNode n;
  n.name = "expand";
  n.inputs = {in};
  n.output = out;
  n.fn = [new_shape](const std::vector<TensorPtr> &ins,
                     const std::unordered_map<std::string, std::string> &) {
    return ins[0]->expand(new_shape);
  };
  n.infer = [new_shape](const std::vector<TensorDesc> &ids,
                        const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 1)
      throw std::runtime_error("expand infer: need input tensor");
    TensorDesc td;
    td.dtype = ids[0].dtype;
    td.shape = new_shape;
    td.strides = Tensor::calc_strides(td.shape);
    return td;
  };
  return n;
}

inline OpNode make_repeat_node(const std::string &in, const std::string &out,
                               const std::vector<int> &repeats) {
  OpNode n;
  n.name = "repeat";
  n.inputs = {in};
  n.output = out;
  n.fn = [repeats](const std::vector<TensorPtr> &ins,
                   const std::unordered_map<std::string, std::string> &) {
    return ins[0]->repeat(repeats);
  };
  n.infer = [repeats](const std::vector<TensorDesc> &ids,
                      const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 1)
      throw std::runtime_error("repeat infer: need input tensor");
    auto input_shape = ids[0].shape;
    if (repeats.size() != input_shape.size()) {
      throw std::runtime_error(
          "repeat infer: repeats size must match input dimensions");
    }
    std::vector<int> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      output_shape.push_back(input_shape[i] * repeats[i]);
    }
    TensorDesc td;
    td.dtype = ids[0].dtype;
    td.shape = output_shape;
    td.strides = Tensor::calc_strides(td.shape);
    return td;
  };
  return n;
}

// -------------------- pow & mean operations --------------------

// pow 操作实现函数（标量指数）
inline TensorPtr pow_scalar(const TensorPtr &X, float exponent) {
  return X->pow(exponent);
}

// pow 操作实现函数（张量指数）
inline TensorPtr pow_tensor(const TensorPtr &X, const TensorPtr &exponent) {
  return X->pow(exponent);
}

// mean 操作实现函数（全局均值）
inline TensorPtr mean_global(const TensorPtr &X) {
  return X->mean();
}

// mean 操作实现函数（指定轴均值）
inline TensorPtr mean_axis(const TensorPtr &X, int axis, bool keepdim = false) {
  return X->mean(axis, keepdim);
}

// 类型转换操作实现函数
inline TensorPtr to_dtype(const TensorPtr &X, DType dtype) {
  return X->to(dtype);
}

// pow 工厂函数（标量指数）
inline OpNode make_pow_scalar_node(const std::string &in,
                                   const std::string &out, float exponent) {
  OpNode n;
  n.name = "pow_scalar";
  n.inputs = {in};
  n.output = out;
  n.fn = [exponent](const std::vector<TensorPtr> &ins,
                    const std::unordered_map<std::string, std::string> &) {
    return pow_scalar(ins[0], exponent);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 1)
      throw std::runtime_error("pow_scalar infer: need input tensor");
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = ids[0].shape;
    td.strides = Tensor::calc_strides(td.shape);
    return td;
  };
  return n;
}

// pow 工厂函数（张量指数）
inline OpNode make_pow_tensor_node(const std::string &in,
                                   const std::string &exponent,
                                   const std::string &out) {
  OpNode n;
  n.name = "pow_tensor";
  n.inputs = {in, exponent};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return pow_tensor(ins[0], ins[1]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 2)
      throw std::runtime_error(
          "pow_tensor infer: need input and exponent tensors");
    auto s = Tensor::broadcast_shape(ids[0].shape, ids[1].shape);
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = s;
    td.strides = Tensor::calc_strides(s);
    return td;
  };
  return n;
}

// mean 工厂函数（全局均值）
inline OpNode make_mean_global_node(const std::string &in,
                                    const std::string &out) {
  OpNode n;
  n.name = "mean_global";
  n.inputs = {in};
  n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return mean_global(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 1)
      throw std::runtime_error("mean_global infer: need input tensor");
    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape.clear(); // 标量张量
    td.strides.clear();
    return td;
  };
  return n;
}

// mean 工厂函数（指定轴均值）
inline OpNode make_mean_axis_node(const std::string &in, const std::string &out,
                                  int axis, bool keepdim = false) {
  OpNode n;
  n.name = "mean_axis";
  n.inputs = {in};
  n.output = out;
  n.fn = [axis, keepdim](const std::vector<TensorPtr> &ins,
                         const std::unordered_map<std::string, std::string> &) {
    return mean_axis(ins[0], axis, keepdim);
  };
  n.infer = [axis,
             keepdim](const std::vector<TensorDesc> &ids,
                      const std::unordered_map<std::string, std::string> &) {
    if (ids.size() < 1)
      throw std::runtime_error("mean_axis infer: need input tensor");
    auto input_shape = ids[0].shape;
    int ndim = static_cast<int>(input_shape.size());
    int norm_axis = axis < 0 ? axis + ndim : axis;
    if (norm_axis < 0 || norm_axis >= ndim) {
      throw std::runtime_error("mean_axis infer: axis out of range");
    }

    // 计算输出形状
    std::vector<int> out_shape;
    if (keepdim) {
      // 保持维度，将指定轴设为1
      for (int i = 0; i < ndim; ++i) {
        if (i == norm_axis) {
          out_shape.push_back(1);
        } else {
          out_shape.push_back(input_shape[i]);
        }
      }
    } else {
      // 移除指定轴
      for (int i = 0; i < ndim; ++i) {
        if (i != norm_axis) {
          out_shape.push_back(input_shape[i]);
        }
      }
      if (out_shape.empty())
        out_shape = {1}; // 如果所有维度都被移除，结果是标量
    }

    TensorDesc td;
    td.dtype = DType::FLOAT32;
    td.shape = out_shape;
    td.strides = Tensor::calc_strides(td.shape);
    return td;
  };
  return n;
}

// 类型转换工厂函数
inline OpNode make_to_dtype_node(const std::string &in, const std::string &out,
                                  DType dtype) {
  OpNode n;
  n.name = "to_dtype";
  n.inputs = {in};
  n.output = out;
  n.fn = [dtype](const std::vector<TensorPtr> &ins,
                 const std::unordered_map<std::string, std::string> &) {
    return to_dtype(ins[0], dtype);
  };
  n.infer = [dtype](const std::vector<TensorDesc> &in_descs,
                    const std::unordered_map<std::string, std::string> &) {
    TensorDesc td = in_descs[0];
    td.dtype = dtype;
    return td;
  };
  return n;
}

// -------------------- NodeRegistry & register_standard_ops
// --------------------
using NodeFactory =
    std::function<OpNode(const std::vector<std::string> &, const std::string &,
                         const std::unordered_map<std::string, std::string> &)>;

struct NodeRegistry {
  std::unordered_map<std::string, NodeFactory> factories;
  static NodeRegistry &instance() {
    static NodeRegistry inst;
    return inst;
  }
  void register_factory(const std::string &op, const NodeFactory &f) {
    factories[op] = f;
  }
  bool has(const std::string &op) const {
    return factories.find(op) != factories.end();
  }
  OpNode make(const std::string &op, const std::vector<std::string> &ins,
              const std::string &out,
              const std::unordered_map<std::string, std::string> &attrs = {}) {
    auto it = factories.find(op);
    if (it == factories.end())
      throw std::runtime_error(std::string("NodeRegistry: unknown op '") + op +
                               "'");
    return it->second(ins, out, attrs);
  }
};

inline void register_standard_ops() {
  static bool inited = false;
  if (inited)
    return;
  inited = true;
  auto &R = NodeRegistry::instance();
  // unary activations
  R.register_factory(
      "relu", [](const std::vector<std::string> &ins, const std::string &out,
                 const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 1)
          throw std::runtime_error("relu needs 1 input");
        return make_relu_node(ins[0], out);
      });
  R.register_factory(
      "sigmoid", [](const std::vector<std::string> &ins, const std::string &out,
                    const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 1)
          throw std::runtime_error("sigmoid needs 1 input");
        return make_sigmoid_node(ins[0], out);
      });
  R.register_factory(
      "tanh", [](const std::vector<std::string> &ins, const std::string &out,
                 const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 1)
          throw std::runtime_error("tanh needs 1 input");
        return make_tanh_node(ins[0], out);
      });
  R.register_factory(
      "gelu", [](const std::vector<std::string> &ins, const std::string &out,
                 const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("gelu needs 1 input");
        bool approx = attr_b(attrs, "approximate", true);
        return make_gelu_node(ins[0], out, approx);
      });
  R.register_factory(
      "softmax", [](const std::vector<std::string> &ins, const std::string &out,
                    const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("softmax needs 1 input");
        int axis = attr_i(attrs, "axis", -1);
        return make_softmax_node(ins[0], out, axis);
      });
  // linear & matmul & elementwise
  R.register_factory(
      "linear", [](const std::vector<std::string> &ins, const std::string &out,
                   const std::unordered_map<std::string, std::string> &) {
        if (ins.size() < 2 || ins.size() > 3)
          throw std::runtime_error("linear needs 2 or 3 inputs");
        return make_linear_node(ins[0], ins[1], out,
                                ins.size() == 3 ? ins[2] : "");
      });
  R.register_factory(
      "matmul", [](const std::vector<std::string> &ins, const std::string &out,
                   const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 2)
          throw std::runtime_error("matmul needs 2 inputs");
        return make_matmul_node(ins[0], ins[1], out);
      });
  R.register_factory(
      "add", [](const std::vector<std::string> &ins, const std::string &out,
                const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 2)
          throw std::runtime_error("add needs 2 inputs");
        return make_add_node(ins[0], ins[1], out);
      });
  R.register_factory(
      "sub", [](const std::vector<std::string> &ins, const std::string &out,
                const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 2)
          throw std::runtime_error("sub needs 2 inputs");
        return make_sub_node(ins[0], ins[1], out);
      });
  R.register_factory(
      "mul", [](const std::vector<std::string> &ins, const std::string &out,
                const std::unordered_map<std::string, std::string> &) {
        if (ins.size() != 2)
          throw std::runtime_error("mul needs 2 inputs");
        return make_mul_node(ins[0], ins[1], out);
      });
  // norm & conv & pooling & reshape & concat
  R.register_factory(
      "batch_norm",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 5)
          throw std::runtime_error("batch_norm needs 5 inputs");
        return make_batch_norm_node(ins[0], ins[1], ins[2], ins[3], ins[4],
                                    out);
      });
  R.register_factory(
      "layer_norm",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 3)
          throw std::runtime_error("layer_norm needs 3 inputs");
        int axis = attr_i(attrs, "axis", -1);
        return make_layer_norm_node(ins[0], ins[1], ins[2], out, axis);
      });
  R.register_factory(
      "conv2d", [](const std::vector<std::string> &ins, const std::string &out,
                   const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() < 2 || ins.size() > 3)
          throw std::runtime_error("conv2d needs 2 or 3 inputs");
        return make_conv2d_node(ins[0], ins[1], out,
                                ins.size() == 3 ? ins[2] : "");
      });
  R.register_factory(
      "avg_pool2d",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("avg_pool2d needs 1 input");
        return make_avg_pool2d_node(ins[0], out);
      });
  R.register_factory(
      "flatten", [](const std::vector<std::string> &ins, const std::string &out,
                    const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("flatten needs 1 input");
        return make_flatten_node(ins[0], out);
      });
  R.register_factory(
      "concat", [](const std::vector<std::string> &ins, const std::string &out,
                   const std::unordered_map<std::string, std::string> &attrs) {
        int axis = attr_i(attrs, "axis", -1);
        return make_concat_node(ins, out, axis);
      });
  R.register_factory(
      "slice", [](const std::vector<std::string> &ins, const std::string &out,
                  const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("slice needs 1 input");
        int axis = attr_i(attrs, "axis", -1);
        int start = attr_i(attrs, "start", 0);
        int length = attr_i(attrs, "length", -1);
        int step = attr_i(attrs, "step", 1);
        return make_slice_node(ins[0], out, axis, start, length, step);
      });
  R.register_factory(
      "slice_view",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("slice_view needs 1 input");
        int axis = attr_i(attrs, "axis", -1);
        int start = attr_i(attrs, "start", 0);
        int length = attr_i(attrs, "length", -1);
        int step = attr_i(attrs, "step", 1);
        return make_slice_view_node(ins[0], out, axis, start, length, step);
      });
  // shape ops
  R.register_factory(
      "shape_of",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_of needs 1 input");
        return make_shape_of_node(ins[0], out);
      });
  R.register_factory(
      "shape_slice",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_slice needs 1 input");
        int start = attr_i(attrs, "start", 0);
        int length = attr_i(attrs, "length", -1);
        int step = attr_i(attrs, "step", 1);
        return make_shape_slice_node(ins[0], out, start, length, step);
      });
  R.register_factory(
      "shape_all_but_last",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_all_but_last needs 1 input");
        return make_shape_all_but_last_node(ins[0], out);
      });
  R.register_factory(
      "shape_all_but_first",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_all_but_first needs 1 input");
        return make_shape_all_but_first_node(ins[0], out);
      });
  R.register_factory(
      "shape_first_n",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_first_n needs 1 input");
        int n = attr_i(attrs, "n", 1);
        return make_shape_first_n_node(ins[0], out, n);
      });
  R.register_factory(
      "shape_last_n",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_last_n needs 1 input");
        int n = attr_i(attrs, "n", 1);
        return make_shape_last_n_node(ins[0], out, n);
      });
  R.register_factory(
      "shape_select",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_select needs 1 input");
        int index = attr_i(attrs, "index", 0);
        return make_shape_select_node(ins[0], out, index);
      });
  R.register_factory(
      "shape_rank",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("shape_rank needs 1 input");
        return make_shape_rank_node(ins[0], out);
      });
  R.register_factory(
      "unsqueeze",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("unsqueeze needs 1 input");
        int axis = attr_i(attrs, "axis", 0);
        return make_unsqueeze_node(ins[0], out, axis);
      });
  R.register_factory(
      "squeeze", [](const std::vector<std::string> &ins, const std::string &out,
                    const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("squeeze needs 1 input");
        bool has_axis = attrs.find("axis") != attrs.end();
        int axis = has_axis ? attr_i(attrs, "axis", 0) : 0;
        return make_squeeze_node(ins[0], out, has_axis, axis);
      });
  R.register_factory(
      "permute", [](const std::vector<std::string> &ins, const std::string &out,
                    const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("permute needs 1 input");
        auto it = attrs.find("dims");
        if (it == attrs.end())
          throw std::runtime_error(
              "permute: missing 'dims' attribute like '0,2,1'");
        std::vector<int> dims;
        dims.reserve(8);
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          if (!tok.empty())
            dims.push_back(std::stoi(tok));
        }
        return make_permute_node(ins[0], out, dims);
      });
  // residual
  R.register_factory(
      "residual_add",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 2)
          throw std::runtime_error("residual_add needs 2 inputs");
        OpNode n;
        n.name = "residual_add";
        n.inputs = ins;
        n.output = out;
        n.fn = [attrs](const std::vector<TensorPtr> &vs,
                       const std::unordered_map<std::string, std::string> &) {
          float alpha = attr_f(attrs, "alpha", 1.0f);
          return residual_add(vs[0], vs[1], alpha);
        };
        n.infer = [](const std::vector<TensorDesc> &ids,
                     const std::unordered_map<std::string, std::string> &) {
          auto td = ids[0];
          td.dtype = DType::FLOAT32;
          return td;
        };
        return n;
      });
  // dropout
  R.register_factory(
      "dropout", [](const std::vector<std::string> &ins, const std::string &out,
                    const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("dropout needs 1 input");
        OpNode n;
        n.name = "dropout";
        n.inputs = ins;
        n.output = out;
        n.fn = [attrs](const std::vector<TensorPtr> &vs,
                       const std::unordered_map<std::string, std::string> &) {
          float p = attr_f(attrs, "p", 0.0f);
          bool training = attr_b(attrs, "training", false);
          unsigned int seed = (unsigned int)attr_i(attrs, "seed", 42);
          return dropout(vs[0], p, training, seed);
        };
        n.infer = [](const std::vector<TensorDesc> &ids,
                     const std::unordered_map<std::string, std::string> &) {
          auto td = ids[0];
          td.dtype = DType::FLOAT32;
          return td;
        };
        return n;
      });
  // expand
  R.register_factory(
      "expand", [](const std::vector<std::string> &ins, const std::string &out,
                   const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("expand needs 1 input");
        auto it = attrs.find("shape");
        if (it == attrs.end())
          throw std::runtime_error(
              "expand: missing 'shape' attribute like '2,3,4'");
        std::vector<int> new_shape;
        new_shape.reserve(8);
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          if (!tok.empty())
            new_shape.push_back(std::stoi(tok));
        }
        return make_expand_node(ins[0], out, new_shape);
      });
  // repeat
  R.register_factory(
      "repeat", [](const std::vector<std::string> &ins, const std::string &out,
                   const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("repeat needs 1 input");
        auto it = attrs.find("repeats");
        if (it == attrs.end())
          throw std::runtime_error(
              "repeat: missing 'repeats' attribute like '2,3,1'");
        std::vector<int> repeats;
        repeats.reserve(8);
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          if (!tok.empty())
            repeats.push_back(std::stoi(tok));
        }
        return make_repeat_node(ins[0], out, repeats);
      });

  // pow_scalar
  R.register_factory(
      "pow_scalar",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("pow_scalar needs 1 input");
        float exponent = attr_f(attrs, "exponent", 1.0f);
        return make_pow_scalar_node(ins[0], out, exponent);
      });

  // pow_tensor
  R.register_factory(
      "pow_tensor",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 2)
          throw std::runtime_error("pow_tensor needs 2 inputs");
        return make_pow_tensor_node(ins[0], ins[1], out);
      });

  // mean_global
  R.register_factory(
      "mean_global",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("mean_global needs 1 input");
        return make_mean_global_node(ins[0], out);
      });

  // mean_axis
  R.register_factory(
      "mean_axis",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("mean_axis needs 1 input");
        int axis = attr_i(attrs, "axis", -1);
        return make_mean_axis_node(ins[0], out, axis);
      });

  // to_dtype
  R.register_factory(
      "to_dtype",
      [](const std::vector<std::string> &ins, const std::string &out,
         const std::unordered_map<std::string, std::string> &attrs) {
        if (ins.size() != 1)
          throw std::runtime_error("to_dtype needs 1 input");
        // 从属性中获取目标数据类型
        auto dtype_str = attrs.find("dtype");
        if (dtype_str == attrs.end())
          throw std::runtime_error("to_dtype needs dtype attribute");
        
        DType dtype;
        if (dtype_str->second == "float32") {
          dtype = DType::FLOAT32;
        } else if (dtype_str->second == "float16") {
          dtype = DType::FP16;
        } else if (dtype_str->second == "int32") {
          dtype = DType::INT32;
        } else if (dtype_str->second == "int8") {
          dtype = DType::INT8;
        } else {
          throw std::runtime_error("Unsupported dtype: " + dtype_str->second);
        }
        
        return make_to_dtype_node(ins[0], out, dtype);
      });
}

// -------------------- 简易 GraphBuilder --------------------
struct SliceOrder {
  int start;
  int length;
  int axis;
  int step;
  SliceOrder(int s, int l, int a, int st = 1)
      : start(s), length(l), axis(a), step(st) {}
};

class GraphBuilder {
  ComputeGraph &g;

public:
  explicit GraphBuilder(ComputeGraph &graph) : g(graph) {
    register_standard_ops();
  }
  GraphBuilder &input(const std::string &name, const TensorPtr &t) {
    g.add_input(name, t);
    return *this;
  }
  GraphBuilder &
  apply(const std::string &op, const std::vector<std::string> &ins,
        const std::string &out,
        const std::unordered_map<std::string, std::string> &attrs = {}) {
    OpNode n = NodeRegistry::instance().make(op, ins, out, attrs);
    g.add_node(n);
    return *this;
  }
  // 便捷方法
  GraphBuilder &linear(const std::string &x, const std::string &w,
                       const std::string &y, const std::string &b = "") {
    return apply("linear",
                 b.empty() ? std::vector<std::string>{x, w}
                           : std::vector<std::string>{x, w, b},
                 y);
  }
  GraphBuilder &matmul(const std::string &a, const std::string &b,
                       const std::string &y) {
    return apply("matmul", {a, b}, y);
  }
  GraphBuilder &add(const std::string &a, const std::string &b,
                    const std::string &y) {
    return apply("add", {a, b}, y);
  }
  GraphBuilder &sub(const std::string &a, const std::string &b,
                    const std::string &y) {
    return apply("sub", {a, b}, y);
  }
  GraphBuilder &mul(const std::string &a, const std::string &b,
                    const std::string &y) {
    return apply("mul", {a, b}, y);
  }
  GraphBuilder &relu(const std::string &in, const std::string &out) {
    return apply("relu", {in}, out);
  }
  GraphBuilder &sigmoid(const std::string &in, const std::string &out) {
    return apply("sigmoid", {in}, out);
  }
  GraphBuilder &tanh(const std::string &in, const std::string &out) {
    return apply("tanh", {in}, out);
  }
  GraphBuilder &gelu(const std::string &in, const std::string &out,
                     bool approximate = true) {
    return apply("gelu", {in}, out,
                 std::unordered_map<std::string, std::string>{
                     {"approximate", approximate ? "1" : "0"}});
  }
  GraphBuilder &softmax(const std::string &in, const std::string &out,
                        int axis = -1) {
    return apply("softmax", {in}, out,
                 std::unordered_map<std::string, std::string>{
                     {"axis", std::to_string(axis)}});
  }
  GraphBuilder &layer_norm(const std::string &x, const std::string &g,
                           const std::string &b, const std::string &y,
                           int axis = -1, float eps = 1e-5f) {
    std::unordered_map<std::string, std::string> attrs{
        {"axis", std::to_string(axis)}, {"eps", std::to_string(eps)}};
    return apply("layer_norm", {x, g, b}, y, attrs);
  }
  GraphBuilder &batch_norm(const std::string &x, const std::string &g,
                           const std::string &b, const std::string &mean,
                           const std::string &var, const std::string &y,
                           float eps = 1e-5f) {
    std::unordered_map<std::string, std::string> attrs{
        {"eps", std::to_string(eps)}};
    return apply("batch_norm", {x, g, b, mean, var}, y, attrs);
  }
  GraphBuilder &conv2d(const std::string &x, const std::string &w,
                       const std::string &y, const std::string &bias = "",
                       int sh = 1, int sw = 1, int ph = 0, int pw = 0,
                       int dh = 1, int dw = 1, int groups = 1) {
    std::unordered_map<std::string, std::string> attrs{
        {"stride_h", std::to_string(sh)},   {"stride_w", std::to_string(sw)},
        {"pad_h", std::to_string(ph)},      {"pad_w", std::to_string(pw)},
        {"dilation_h", std::to_string(dh)}, {"dilation_w", std::to_string(dw)},
        {"groups", std::to_string(groups)}};
    return apply("conv2d",
                 bias.empty() ? std::vector<std::string>{x, w}
                              : std::vector<std::string>{x, w, bias},
                 y, attrs);
  }
  GraphBuilder &flatten(const std::string &in, const std::string &out,
                        int start_axis = 1, int end_axis = -1) {
    std::unordered_map<std::string, std::string> attrs{
        {"start_axis", std::to_string(start_axis)},
        {"end_axis", std::to_string(end_axis)}};
    return apply("flatten", {in}, out, attrs);
  }
  GraphBuilder &unsqueeze(const std::string &in, const std::string &out,
                          int axis) {
    std::unordered_map<std::string, std::string> attrs{
        {"axis", std::to_string(axis)}};
    return apply("unsqueeze", {in}, out, attrs);
  }
  GraphBuilder &squeeze(const std::string &in, const std::string &out,
                        int axis) {
    std::unordered_map<std::string, std::string> attrs{
        {"axis", std::to_string(axis)}};
    return apply("squeeze", {in}, out, attrs);
  }
  GraphBuilder &squeeze_all(const std::string &in, const std::string &out) {
    return apply("squeeze", {in}, out);
  }
  GraphBuilder &permute(const std::string &in, const std::string &out,
                        const std::vector<int> &dims) {
    std::ostringstream oss;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i)
        oss << ",";
      oss << dims[i];
    }
    std::unordered_map<std::string, std::string> attrs{{"dims", oss.str()}};
    return apply("permute", {in}, out, attrs);
  }
  GraphBuilder &concat(const std::vector<std::string> &ins,
                       const std::string &out, int axis) {
    std::unordered_map<std::string, std::string> attrs{
        {"axis", std::to_string(axis)}};
    return apply("concat", ins, out, attrs);
  }
  GraphBuilder &slice(const std::string &in, const std::string &out, int axis,
                      int start, int length, int step = 1) {
    std::unordered_map<std::string, std::string> attrs{
        {"axis", std::to_string(axis)},
        {"start", std::to_string(start)},
        {"length", std::to_string(length)},
        {"step", std::to_string(step)}};
    return apply("slice", {in}, out, attrs);
  }
  // Overload mapping (start, length, axis) to existing implementation via
  // SliceOrder
  GraphBuilder &slice(const std::string &in, const std::string &out,
                      const SliceOrder &order) {
    return slice(in, out, order.axis, order.start, order.length, order.step);
  }
  // View-based slicing node
  GraphBuilder &slice_view(const std::string &in, const std::string &out,
                           int axis, int start, int length, int step = 1) {
    std::unordered_map<std::string, std::string> attrs{
        {"axis", std::to_string(axis)},
        {"start", std::to_string(start)},
        {"length", std::to_string(length)},
        {"step", std::to_string(step)}};
    return apply("slice_view", {in}, out, attrs);
  }
  GraphBuilder &slice_view(const std::string &in, const std::string &out,
                           const SliceOrder &order) {
    return slice_view(in, out, order.axis, order.start, order.length,
                      order.step);
  }
  // Shape helpers
  GraphBuilder &shape_of(const std::string &in, const std::string &out) {
    return apply("shape_of", {in}, out);
  }
  GraphBuilder &shape_slice(const std::string &in, const std::string &out,
                            int start, int length, int step = 1) {
    std::unordered_map<std::string, std::string> attrs{
        {"start", std::to_string(start)},
        {"length", std::to_string(length)},
        {"step", std::to_string(step)}};
    return apply("shape_slice", {in}, out, attrs);
  }
  GraphBuilder &shape_all_but_last(const std::string &in,
                                   const std::string &out) {
    return apply("shape_all_but_last", {in}, out);
  }
  GraphBuilder &shape_all_but_first(const std::string &in,
                                    const std::string &out) {
    return apply("shape_all_but_first", {in}, out);
  }
  GraphBuilder &shape_first_n(const std::string &in, const std::string &out,
                              int n) {
    std::unordered_map<std::string, std::string> attrs{{"n", std::to_string(n)}};
    return apply("shape_first_n", {in}, out, attrs);
  }
  GraphBuilder &shape_last_n(const std::string &in, const std::string &out,
                             int n) {
    std::unordered_map<std::string, std::string> attrs{{"n", std::to_string(n)}};
    return apply("shape_last_n", {in}, out, attrs);
  }
  GraphBuilder &shape_select(const std::string &in, const std::string &out,
                             int index) {
    std::unordered_map<std::string, std::string> attrs{{"index", std::to_string(index)}};
    return apply("shape_select", {in}, out, attrs);
  }
  GraphBuilder &shape_rank(const std::string &in, const std::string &out) {
    return apply("shape_rank", {in}, out);
  }
  GraphBuilder &avg_pool2d(const std::string &in, const std::string &out,
                           int kh = 2, int kw = 2, int sh = 2, int sw = 2,
                           int ph = 0, int pw = 0, int dh = 1, int dw = 1) {
    std::unordered_map<std::string, std::string> attrs{
        {"kernel_h", std::to_string(kh)},   {"kernel_w", std::to_string(kw)},
        {"stride_h", std::to_string(sh)},   {"stride_w", std::to_string(sw)},
        {"pad_h", std::to_string(ph)},      {"pad_w", std::to_string(pw)},
        {"dilation_h", std::to_string(dh)}, {"dilation_w", std::to_string(dw)}};
    return apply("avg_pool2d", {in}, out, attrs);
  }
  GraphBuilder &residual_add(const std::string &a, const std::string &b,
                             const std::string &out, float alpha = 1.0f) {
    std::unordered_map<std::string, std::string> attrs{
        {"alpha", std::to_string(alpha)}};
    return apply("residual_add", {a, b}, out, attrs);
  }
  GraphBuilder &dropout(const std::string &in, const std::string &out, float p,
                        bool training = false, unsigned int seed = 42) {
    std::unordered_map<std::string, std::string> attrs{
        {"p", std::to_string(p)},
        {"training", training ? "1" : "0"},
        {"seed", std::to_string(seed)}};
    return apply("dropout", {in}, out, attrs);
  }
  // 运行前校验
  void validate_and_run() {
    g.validate();
    g.run();
  }
};

} // namespace ow::nn