#pragma once
#include "tensor.hpp"
#include "cgraph.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ow::nn {

// -------------------- helpers --------------------
static inline std::shared_ptr<Context> get_ctx_or_throw(const TensorPtr &T) {
  auto ctx = T->ctx.lock();
  if (!ctx) throw std::runtime_error("ctx expired");
  return ctx;
}

static inline int attr_i(const std::unordered_map<std::string, std::string> &attrs,
                         const std::string &key, int defv) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return defv;
  return std::stoi(it->second);
}
static inline float attr_f(const std::unordered_map<std::string, std::string> &attrs,
                           const std::string &key, float defv) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return defv;
  return std::stof(it->second);
}
static inline bool attr_b(const std::unordered_map<std::string, std::string> &attrs,
                          const std::string &key, bool defv) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return defv;
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
  if (ax < 0 || ax >= int(r)) throw std::runtime_error("softmax axis out of range");
  int dim = X->shape[ax];
  size_t outer = 1; for (int i = 0; i < ax; ++i) outer *= (size_t)X->shape[i];
  size_t inner = 1; for (size_t i = ax + 1; i < r; ++i) inner *= (size_t)X->shape[i];
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

// -------------------- ComputeGraph node factories (activations) --------------------
inline OpNode make_relu_node(const std::string &in, const std::string &out) {
  OpNode n; n.name = "relu"; n.inputs = {in}; n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return relu(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- BatchNorm --------------------
inline TensorPtr batch_norm(const TensorPtr &X,
                            const TensorPtr &gamma,
                            const TensorPtr &beta,
                            const TensorPtr &running_mean,
                            const TensorPtr &running_var,
                            float eps = 1e-5f) {
  if (X->shape.size() != 4 && X->shape.size() != 3)
    throw std::runtime_error("batch_norm expects rank 3 or 4 (CHW/NCHW)");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("batch_norm requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int N = 1, C, H, W;
  if (X->shape.size() == 4) { N = X->shape[0]; C = X->shape[1]; H = X->shape[2]; W = X->shape[3]; }
  else { C = X->shape[0]; H = X->shape[1]; W = X->shape[2]; }
  if ((int)gamma->shape.size() != 1 || (int)beta->shape.size() != 1 ||
      (int)running_mean->shape.size() != 1 || (int)running_var->shape.size() != 1)
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
            size_t idx = (size_t)n * XcW + (size_t)c * XinW + (size_t)h * (size_t)W + (size_t)w;
            float x = X->get_as_float_flat(idx);
            float y = (x - m) * invstd * g + b;
            size_t oidx = (size_t)n * YcW + (size_t)c * YonW + (size_t)h * (size_t)W + (size_t)w;
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
          size_t idx = (size_t)c * (size_t)H * (size_t)W + (size_t)h * (size_t)W + (size_t)w;
          float x = X->get_as_float_flat(idx);
          float y = (x - m) * invstd * g + b;
          size_t oidx = (size_t)c * (size_t)H * (size_t)W + (size_t)h * (size_t)W + (size_t)w;
          Y->set_from_float_flat(oidx, y);
        }
      }
    }
  }
  return Y;
}

inline OpNode make_batch_norm_node(const std::string &in, const std::string &gamma,
                                   const std::string &beta, const std::string &mean,
                                   const std::string &var, const std::string &out) {
  OpNode n; n.name = "batch_norm"; n.inputs = {in, gamma, beta, mean, var}; n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    float eps = attr_f(attrs, "eps", 1e-5f);
    return batch_norm(ins[0], ins[1], ins[2], ins[3], ins[4], eps);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    auto td = ids[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- LayerNorm --------------------
inline TensorPtr layer_norm(const TensorPtr &X,
                            const TensorPtr &gamma,
                            const TensorPtr &beta,
                            float eps = 1e-5f, int axis = -1) {
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("layer_norm requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int r = (int)X->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r) throw std::runtime_error("layer_norm axis out of range");
  int dim = X->shape[ax];
  if ((int)gamma->shape.size() != 1 || (int)beta->shape.size() != 1 ||
      gamma->shape[0] != dim || beta->shape[0] != dim)
    throw std::runtime_error("layer_norm: gamma/beta size mismatch with axis dim");
  size_t outer = 1; for (int i = 0; i < ax; ++i) outer *= (size_t)X->shape[i];
  size_t inner = 1; for (int i = ax + 1; i < r; ++i) inner *= (size_t)X->shape[i];
  auto Y = Tensor::create(ctx, X->shape, DType::FLOAT32);
  size_t seg_len = (size_t)dim * inner;
  for (size_t o = 0; o < outer; ++o) {
    size_t base = o * seg_len;
    for (size_t in = 0; in < inner; ++in) {
      float mean = 0.0f;
      for (int j = 0; j < dim; ++j) mean += X->get_as_float_flat(base + (size_t)j * inner + in);
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

inline OpNode make_layer_norm_node(const std::string &in, const std::string &gamma,
                                   const std::string &beta, const std::string &out,
                                   int axis = -1) {
  OpNode n; n.name = "layer_norm"; n.inputs = {in, gamma, beta}; n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &ins,
                const std::unordered_map<std::string, std::string> &attrs) {
    float eps = attr_f(attrs, "eps", 1e-5f);
    int ax = axis;
    if (attrs.find("axis") != attrs.end()) ax = attr_i(attrs, "axis", axis);
    return layer_norm(ins[0], ins[1], ins[2], eps, ax);
  };
  n.infer = [axis](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    auto td = ids[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- Dropout --------------------
inline TensorPtr dropout(const TensorPtr &X, float p, bool training = true, unsigned int seed = 42) {
  if (p < 0.0f || p >= 1.0f) throw std::runtime_error("dropout: p in [0,1)");
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
  OpNode n; n.name = "dropout"; n.inputs = {in}; n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    float p = attr_f(attrs, "p", 0.0f);
    bool training = attr_b(attrs, "training", false);
    unsigned int seed = (unsigned int)attr_i(attrs, "seed", 42);
    return dropout(ins[0], p, training, seed);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    auto td = ids[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- Residual --------------------
inline TensorPtr residual_add(const TensorPtr &A, const TensorPtr &B, float alpha = 1.0f) {
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

inline OpNode make_residual_node(const std::string &a, const std::string &b, const std::string &out) {
  OpNode n; n.name = "residual_add"; n.inputs = {a, b}; n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &attrs) {
    float alpha = attr_f(attrs, "alpha", 1.0f);
    return residual_add(ins[0], ins[1], alpha);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &) {
    // broadcast infer: use helper from Tensor if needed; for simplicity assume same shape
    auto td = ids[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- Conv2d (NCHW/CHW) --------------------
inline TensorPtr conv2d(const TensorPtr &X,
                        const TensorPtr &W,
                        const TensorPtr &bias,
                        int sh = 1, int sw = 1,
                        int ph = 0, int pw = 0,
                        int dh = 1, int dw = 1,
                        int groups = 1) {
  if (!X->is_contiguous_row_major() || !W->is_contiguous_row_major())
    throw std::runtime_error("conv2d requires contiguous inputs");
  if (groups != 1) throw std::runtime_error("conv2d: groups>1 not implemented");
  if ((int)W->shape.size() != 4) throw std::runtime_error("conv2d: W shape [Cout,Cin,Kh,Kw]");
  int N = 1, C, H, W_in;
  if (X->shape.size() == 4) { N = X->shape[0]; C = X->shape[1]; H = X->shape[2]; W_in = X->shape[3]; }
  else if (X->shape.size() == 3) { C = X->shape[0]; H = X->shape[1]; W_in = X->shape[2]; }
  else throw std::runtime_error("conv2d expects rank 3 or 4");
  int Cout = W->shape[0];
  int Cin = W->shape[1];
  int Kh = W->shape[2];
  int Kw = W->shape[3];
  if (Cin != C) throw std::runtime_error("conv2d: Cin mismatch");
  if (bias && (int)bias->shape.size() != 1) throw std::runtime_error("conv2d: bias must be 1D");
  if (bias && bias->shape[0] != Cout) throw std::runtime_error("conv2d: bias size mismatch");
  int eff_kh = dh * (Kh - 1) + 1;
  int eff_kw = dw * (Kw - 1) + 1;
  int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
  int Wo = std::max(0, (W_in + 2 * pw - eff_kw) / sw + 1);
  std::vector<int> out_shape = (X->shape.size() == 4) ? std::vector<int>{N, Cout, Ho, Wo}
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
                    size_t xidx = (size_t)n * XcW + (size_t)ic * XinW + (size_t)ih * (size_t)W_in + (size_t)iw;
                    size_t widx = (size_t)oc * WcW + (size_t)ic * WhW + (size_t)kh_i * (size_t)Kw + (size_t)kw_i;
                    acc += X->get_as_float_flat(xidx) * W->get_as_float_flat(widx);
                  }
                }
              }
            }
            size_t yidx = (size_t)n * YcW + (size_t)oc * YonW + (size_t)oh * (size_t)Wo + (size_t)ow;
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
                  size_t xidx = (size_t)ic * XinW + (size_t)ih * (size_t)W_in + (size_t)iw;
                  size_t widx = (size_t)oc * WcW + (size_t)ic * WhW + (size_t)kh_i * (size_t)Kw + (size_t)kw_i;
                  acc += X->get_as_float_flat(xidx) * W->get_as_float_flat(widx);
                }
              }
            }
          }
          size_t yidx = (size_t)oc * (size_t)Ho * (size_t)Wo + (size_t)oh * (size_t)Wo + (size_t)ow;
          Y->set_from_float_flat(yidx, acc);
        }
      }
    }
  }
  return Y;
}

inline OpNode make_conv2d_node(const std::string &in, const std::string &weight,
                               const std::string &out, const std::string &bias_optional = "") {
  OpNode n; n.name = "conv2d"; n.inputs = bias_optional.empty() ? std::vector<std::string>{in, weight}
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
    const TensorPtr &X = ins[0]; const TensorPtr &W = ins[1];
    const TensorPtr bias = (ins.size() >= 3) ? ins[2] : nullptr;
    return conv2d(X, W, bias, sh, sw, ph, pw, dh, dw, groups);
  };
  n.infer = [](const std::vector<TensorDesc> &ids,
               const std::unordered_map<std::string, std::string> &attrs) {
    if (ids.size() < 2) throw std::runtime_error("conv2d infer: need X and W");
    auto Xd = ids[0]; auto Wd = ids[1];
    if ((int)Wd.shape.size() != 4) throw std::runtime_error("conv2d infer: W rank 4");
    int Cout = Wd.shape[0]; int Kh = Wd.shape[2]; int Kw = Wd.shape[3];
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
      Xd.shape = {Xd.shape[0], Cout, Ho, Wo}; Xd.dtype = DType::FLOAT32; return Xd;
    } else if (Xd.shape.size() == 3) {
      int H = Xd.shape[1], W = Xd.shape[2];
      int eff_kh = dh * (Kh - 1) + 1;
      int eff_kw = dw * (Kw - 1) + 1;
      int Ho = std::max(0, (H + 2 * ph - eff_kh) / sh + 1);
      int Wo = std::max(0, (W + 2 * pw - eff_kw) / sw + 1);
      Xd.shape = {Cout, Ho, Wo}; Xd.dtype = DType::FLOAT32; return Xd;
    } else {
      throw std::runtime_error("conv2d infer: X rank 3 or 4");
    }
  };
  return n;
}

inline OpNode make_sigmoid_node(const std::string &in, const std::string &out) {
  OpNode n; n.name = "sigmoid"; n.inputs = {in}; n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return sigmoid(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

inline OpNode make_tanh_node(const std::string &in, const std::string &out) {
  OpNode n; n.name = "tanh"; n.inputs = {in}; n.output = out;
  n.fn = [](const std::vector<TensorPtr> &ins,
            const std::unordered_map<std::string, std::string> &) {
    return tanh_act(ins[0]);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

inline OpNode make_gelu_node(const std::string &in, const std::string &out, bool approximate = true) {
  OpNode n; n.name = approximate ? "gelu_approx" : "gelu_exact"; n.inputs = {in}; n.output = out;
  n.fn = [approximate](const std::vector<TensorPtr> &ins,
                       const std::unordered_map<std::string, std::string> &) {
    return gelu(ins[0], approximate);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

inline OpNode make_softmax_node(const std::string &in, const std::string &out, int axis = -1) {
  OpNode n; n.name = "softmax"; n.inputs = {in}; n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &ins,
                const std::unordered_map<std::string, std::string> &) {
    return softmax(ins[0], axis);
  };
  n.infer = [](const std::vector<TensorDesc> &in_descs,
               const std::unordered_map<std::string, std::string> &) {
    auto td = in_descs[0]; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- pooling --------------------
inline TensorPtr max_pool2d(const TensorPtr &X,
                            int kh, int kw,
                            int sh = 1, int sw = 1,
                            int ph = 0, int pw = 0,
                            int dh = 1, int dw = 1) {
  if (X->shape.size() != 4 && X->shape.size() != 3)
    throw std::runtime_error("max_pool2d expects rank 3 or 4 (CHW/NCHW)");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("max_pool2d requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int N = 1, C, H, W;
  if (X->shape.size() == 4) { N = X->shape[0]; C = X->shape[1]; H = X->shape[2]; W = X->shape[3]; }
  else { C = X->shape[0]; H = X->shape[1]; W = X->shape[2]; }
  int eff_kh = dh * (kh - 1) + 1;
  int eff_kw = dw * (kw - 1) + 1;
  int Ho = (H + 2 * ph - eff_kh) / sh + 1; Ho = std::max(0, Ho);
  int Wo = (W + 2 * pw - eff_kw) / sw + 1; Wo = std::max(0, Wo);
  std::vector<int> out_shape = (X->shape.size() == 4) ? std::vector<int>{N, C, Ho, Wo}
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
                  size_t idx = (size_t)n * XcW + (size_t)c * XinW + (size_t)ih * (size_t)W + (size_t)iw;
                  best = std::max(best, X->get_as_float_flat(idx));
                }
              }
            }
            size_t oidx = (size_t)n * YcW + (size_t)c * YonW + (size_t)oh * (size_t)Wo + (size_t)ow;
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
                size_t idx = (size_t)c * (size_t)H * (size_t)W + (size_t)ih * (size_t)W + (size_t)iw;
                best = std::max(best, X->get_as_float_flat(idx));
              }
            }
          }
          size_t oidx = (size_t)c * (size_t)Ho * (size_t)Wo + (size_t)oh * (size_t)Wo + (size_t)ow;
          Y->set_from_float_flat(oidx, best);
        }
      }
    }
  }
  return Y;
}

inline TensorPtr avg_pool2d(const TensorPtr &X,
                            int kh, int kw,
                            int sh = 1, int sw = 1,
                            int ph = 0, int pw = 0,
                            int dh = 1, int dw = 1) {
  if (X->shape.size() != 4 && X->shape.size() != 3)
    throw std::runtime_error("avg_pool2d expects rank 3 or 4 (CHW/NCHW)");
  if (!X->is_contiguous_row_major())
    throw std::runtime_error("avg_pool2d requires contiguous tensor");
  auto ctx = get_ctx_or_throw(X);
  int N = 1, C, H, W;
  if (X->shape.size() == 4) { N = X->shape[0]; C = X->shape[1]; H = X->shape[2]; W = X->shape[3]; }
  else { C = X->shape[0]; H = X->shape[1]; W = X->shape[2]; }
  int eff_kh = dh * (kh - 1) + 1;
  int eff_kw = dw * (kw - 1) + 1;
  int Ho = (H + 2 * ph - eff_kh) / sh + 1; Ho = std::max(0, Ho);
  int Wo = (W + 2 * pw - eff_kw) / sw + 1; Wo = std::max(0, Wo);
  std::vector<int> out_shape = (X->shape.size() == 4) ? std::vector<int>{N, C, Ho, Wo}
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
            float sum = 0.0f; int count = 0;
            for (int kh_i = 0; kh_i < kh; ++kh_i) {
              for (int kw_i = 0; kw_i < kw; ++kw_i) {
                int ih = oh * sh - ph + kh_i * dh;
                int iw = ow * sw - pw + kw_i * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  size_t idx = (size_t)n * XcW + (size_t)c * XinW + (size_t)ih * (size_t)W + (size_t)iw;
                  sum += X->get_as_float_flat(idx); ++count;
                }
              }
            }
            float avg = (count > 0) ? (sum / (float)count) : 0.0f;
            size_t oidx = (size_t)n * YcW + (size_t)c * YonW + (size_t)oh * (size_t)Wo + (size_t)ow;
            Y->set_from_float_flat(oidx, avg);
          }
        }
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < Ho; ++oh) {
        for (int ow = 0; ow < Wo; ++ow) {
          float sum = 0.0f; int count = 0;
          for (int kh_i = 0; kh_i < kh; ++kh_i) {
            for (int kw_i = 0; kw_i < kw; ++kw_i) {
              int ih = oh * sh - ph + kh_i * dh;
              int iw = ow * sw - pw + kw_i * dw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                size_t idx = (size_t)c * (size_t)H * (size_t)W + (size_t)ih * (size_t)W + (size_t)iw;
                sum += X->get_as_float_flat(idx); ++count;
              }
            }
          }
          float avg = (count > 0) ? (sum / (float)count) : 0.0f;
          size_t oidx = (size_t)c * (size_t)Ho * (size_t)Wo + (size_t)oh * (size_t)Wo + (size_t)ow;
          Y->set_from_float_flat(oidx, avg);
        }
      }
    }
  }
  return Y;
}

inline OpNode make_max_pool2d_node(const std::string &in, const std::string &out) {
  OpNode n; n.name = "max_pool2d"; n.inputs = {in}; n.output = out;
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
    td.shape[H_idx] = Ho; td.shape[W_idx] = Wo; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

inline OpNode make_avg_pool2d_node(const std::string &in, const std::string &out) {
  OpNode n; n.name = "avg_pool2d"; n.inputs = {in}; n.output = out;
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
    td.shape[H_idx] = Ho; td.shape[W_idx] = Wo; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- flatten --------------------
inline TensorPtr flatten(const TensorPtr &X, int start_axis = 0, int end_axis = -1) {
  int r = (int)X->shape.size();
  int sa = start_axis < 0 ? r + start_axis : start_axis;
  int ea = end_axis < 0 ? r + end_axis : end_axis;
  if (sa < 0 || ea >= r || sa > ea) throw std::runtime_error("flatten: axis out of range");
  std::vector<int> ns;
  ns.reserve(r - (ea - sa));
  for (int i = 0; i < sa; ++i) ns.push_back(X->shape[i]);
  int merged = 1; for (int i = sa; i <= ea; ++i) merged *= X->shape[i];
  ns.push_back(merged);
  for (int i = ea + 1; i < r; ++i) ns.push_back(X->shape[i]);
  if (X->is_contiguous_row_major()) return X->reshape_view(ns);
  auto Xc = X->copy(); return Xc->reshape_view(ns);
}

inline OpNode make_flatten_node(const std::string &in, const std::string &out) {
  OpNode n; n.name = "flatten"; n.inputs = {in}; n.output = out;
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
    if (attrs.find("end_axis") != attrs.end()) ea = attr_i(attrs, "end_axis", -1);
    int sa_n = sa < 0 ? r + sa : sa;
    int ea_n = ea < 0 ? r + ea : ea;
    if (sa_n < 0 || ea_n >= r || sa_n > ea_n) throw std::runtime_error("flatten infer: axis out of range");
    std::vector<int> ns; ns.reserve(r - (ea_n - sa_n));
    for (int i = 0; i < sa_n; ++i) ns.push_back(td.shape[i]);
    int merged = 1; for (int i = sa_n; i <= ea_n; ++i) merged *= td.shape[i];
    ns.push_back(merged);
    for (int i = ea_n + 1; i < r; ++i) ns.push_back(td.shape[i]);
    td.shape = ns; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

// -------------------- concat --------------------
inline TensorPtr concat(const std::vector<TensorPtr> &inputs, int axis) {
  if (inputs.empty()) throw std::runtime_error("concat: empty inputs");
  int r = (int)inputs[0]->shape.size();
  int ax = axis < 0 ? r + axis : axis;
  if (ax < 0 || ax >= r) throw std::runtime_error("concat: axis out of range");
  for (auto &t : inputs) {
    if ((int)t->shape.size() != r) throw std::runtime_error("concat: rank mismatch");
    if (!t->is_contiguous_row_major()) throw std::runtime_error("concat requires contiguous inputs");
  }
  std::vector<int> out_shape = inputs[0]->shape;
  int sum_ax = 0;
  for (auto &t : inputs) {
    for (int i = 0; i < r; ++i) {
      if (i == ax) continue;
      if (t->shape[i] != out_shape[i]) throw std::runtime_error("concat: non-axis dims mismatch");
    }
    sum_ax += t->shape[ax];
  }
  out_shape[ax] = sum_ax;
  auto ctx = get_ctx_or_throw(inputs[0]);
  auto Y = Tensor::create(ctx, out_shape, DType::FLOAT32);
  size_t outer = 1; for (int i = 0; i < ax; ++i) outer *= (size_t)out_shape[i];
  size_t inner = 1; for (int i = ax + 1; i < r; ++i) inner *= (size_t)out_shape[i];
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

inline OpNode make_concat_node(const std::vector<std::string> &ins, const std::string &out, int axis) {
  OpNode n; n.name = "concat"; n.inputs = ins; n.output = out;
  n.fn = [axis](const std::vector<TensorPtr> &vs,
                const std::unordered_map<std::string, std::string> &) {
    return concat(vs, axis);
  };
  n.infer = [axis](const std::vector<TensorDesc> &ids,
                   const std::unordered_map<std::string, std::string> &) {
    if (ids.empty()) throw std::runtime_error("concat infer: empty inputs");
    int r = (int)ids[0].shape.size();
    int ax = axis < 0 ? r + axis : axis;
    if (ax < 0 || ax >= r) throw std::runtime_error("concat infer: axis out of range");
    TensorDesc td = ids[0];
    int sum_ax = 0;
    for (auto &d : ids) {
      if ((int)d.shape.size() != r) throw std::runtime_error("concat infer: rank mismatch");
      for (int i = 0; i < r; ++i) {
        if (i == ax) continue;
        if (d.shape[i] != td.shape[i]) throw std::runtime_error("concat infer: non-axis dims mismatch");
      }
      sum_ax += d.shape[ax];
    }
    td.shape[ax] = sum_ax; td.dtype = DType::FLOAT32; return td;
  };
  return n;
}

} // namespace ow::nn