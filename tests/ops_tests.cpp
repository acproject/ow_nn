#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/backend_qwen3vl.hpp"
#include "../src/ops.hpp"
#include "../src/tensor.hpp"
#include "../src/utils.hpp"

using namespace ow::nn;
namespace fs = std::filesystem;

struct HostTensor {
  std::vector<int> shape;
  std::vector<float> data;
};

static bool read_tensor_bin(const fs::path &path, HostTensor &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "[read] cannot open: " << path << std::endl;
    return false;
  }
  int32_t rank = 0;
  f.read(reinterpret_cast<char *>(&rank), sizeof(int32_t));
  if (!f.good()) {
    std::cerr << "[read] failed reading rank: " << path << std::endl;
    return false;
  }
  out.shape.resize(rank);
  size_t ne = 1;
  for (int i = 0; i < rank; ++i) {
    int32_t d = 0;
    f.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
    if (!f.good()) {
      std::cerr << "[read] failed reading dim " << i << ": " << path
                << std::endl;
      return false;
    }
    out.shape[i] = d;
    ne *= (size_t)d;
  }
  out.data.resize(ne);
  f.read(reinterpret_cast<char *>(out.data.data()), ne * sizeof(float));
  if (!f.good()) {
    std::cerr << "[read] failed reading data: " << path << std::endl;
    return false;
  }
  return true;
}

static TensorPtr make_tensor(const std::shared_ptr<Context> &ctx,
                             const std::vector<int> &shape,
                             const std::vector<float> &data) {
  auto t = Tensor::create(ctx, shape, DType::FLOAT32);
  size_t n = t->nelements();
  if (n != data.size()) {
    throw std::runtime_error("make_tensor size mismatch");
  }
  for (size_t i = 0; i < n; ++i) {
    t->set_from_float_flat(i, data[i]);
  }
  return t;
}

static float max_abs_diff(const TensorPtr &A, const TensorPtr &B) {
  if (!A || !B)
    throw std::runtime_error("diff: null tensor");
  if (A->shape != B->shape)
    throw std::runtime_error("diff: shape mismatch");
  size_t n = A->nelements();
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float da = A->get_as_float_flat(i);
    float db = B->get_as_float_flat(i);
    float d = std::fabs(da - db);
    if (d > m)
      m = d;
  }
  return m;
}

static bool test_linear(const std::shared_ptr<Context> &ctx,
                        const fs::path &dir) {
  HostTensor Xh, Wh, bh, Yh;
  if (!read_tensor_bin(dir / "linear_X.tensor", Xh))
    return false;
  if (!read_tensor_bin(dir / "linear_W.tensor", Wh))
    return false;
  if (!read_tensor_bin(dir / "linear_b.tensor", bh))
    return false;
  if (!read_tensor_bin(dir / "linear_Y.tensor", Yh))
    return false;
  auto X = make_tensor(ctx, Xh.shape, Xh.data);
  auto W = make_tensor(ctx, Wh.shape, Wh.data);
  auto b = make_tensor(ctx, bh.shape, bh.data);
  auto Y_ref = make_tensor(ctx, Yh.shape, Yh.data);
  auto Y = Tensor::linear(X, W, b);
  float mad = max_abs_diff(Y, Y_ref);
  std::cout << "[linear] max_abs_diff=" << mad << std::endl;
  return mad < 1e-5f;
}

static bool test_activations(const std::shared_ptr<Context> &ctx,
                             const fs::path &dir) {
  HostTensor Xin, relu_h, sig_h, tanh_h, gelu_exact_h, gelu_tanh_h, silu_h,
      soft_h;
  if (!read_tensor_bin(dir / "act_input.tensor", Xin))
    return false;
  auto X = make_tensor(ctx, Xin.shape, Xin.data);

  bool ok = true;
  auto Y_relu = Tensor::relu(X);
  read_tensor_bin(dir / "relu.tensor", relu_h);
  ok &= (max_abs_diff(Y_relu, make_tensor(ctx, relu_h.shape, relu_h.data)) <
         1e-5f);

  auto Y_sig = Tensor::sigmoid(X);
  read_tensor_bin(dir / "sigmoid.tensor", sig_h);
  ok &=
      (max_abs_diff(Y_sig, make_tensor(ctx, sig_h.shape, sig_h.data)) < 1e-5f);

  auto Y_tanh = Tensor::tanh_act(X);
  read_tensor_bin(dir / "tanh.tensor", tanh_h);
  ok &= (max_abs_diff(Y_tanh, make_tensor(ctx, tanh_h.shape, tanh_h.data)) <
         1e-5f);

  auto Y_gelu_exact = Tensor::gelu(X, false);
  read_tensor_bin(dir / "gelu_exact.tensor", gelu_exact_h);
  ok &= (max_abs_diff(Y_gelu_exact, make_tensor(ctx, gelu_exact_h.shape,
                                                gelu_exact_h.data)) < 1e-5f);

  auto Y_gelu_tanh = Tensor::gelu(X, true);
  read_tensor_bin(dir / "gelu_tanh.tensor", gelu_tanh_h);
  ok &= (max_abs_diff(Y_gelu_tanh, make_tensor(ctx, gelu_tanh_h.shape,
                                               gelu_tanh_h.data)) < 1e-5f);

  auto Y_silu = Tensor::silu(X);
  read_tensor_bin(dir / "silu.tensor", silu_h);
  ok &= (max_abs_diff(Y_silu, make_tensor(ctx, silu_h.shape, silu_h.data)) <
         1e-5f);

  auto Y_soft = Tensor::softmax(X, -1);
  read_tensor_bin(dir / "softmax.tensor", soft_h);
  ok &= (max_abs_diff(Y_soft, make_tensor(ctx, soft_h.shape, soft_h.data)) <
         1e-5f);

  std::cout << "[activations] all_ok=" << (ok ? "true" : "false") << std::endl;
  return ok;
}

static bool test_rmsnorm(const std::shared_ptr<Context> &ctx,
                         const fs::path &dir) {
  HostTensor Xh, Wh, Yh;
  if (!read_tensor_bin(dir / "rms_X.tensor", Xh))
    return false;
  if (!read_tensor_bin(dir / "rms_weight.tensor", Wh))
    return false;
  if (!read_tensor_bin(dir / "rms_Y.tensor", Yh))
    return false;
  auto X = make_tensor(ctx, Xh.shape, Xh.data);
  auto W = make_tensor(ctx, Wh.shape, Wh.data);
  auto Y_ref = make_tensor(ctx, Yh.shape, Yh.data);

  // Y = (X / sqrt(mean(x^2)+eps)) * weight
  float eps = 1e-6f;
  auto var = X->pow(2.0f)->mean(-1, true);
  auto rsqrt = var->elementwise_unary(
      var, [eps](float v) { return 1.0f / std::sqrt(v + eps); });
  auto Xn = X->tensor_mul(X, rsqrt);
  auto Y = Xn->tensor_mul(Xn, W);

  float mad = max_abs_diff(Y, Y_ref);
  std::cout << "[rmsnorm] max_abs_diff=" << mad << std::endl;
  return mad < 1e-5f;
}

static TensorPtr adapt_rope_cache_to_BSD(const TensorPtr &raw, int B) {
  // Accept shapes: [S,1,D] or [B,S,1,D] or already [B,S,D]
  if (raw->shape.size() == 3) {
    int S = raw->shape[0];
    int one = raw->shape[1];
    int D = raw->shape[2];
    if (one == 1) {
      // [S,1,D] -> [1,S,1,D] -> squeeze axis=2 -> [1,S,D] -> repeat B
      auto u0 = unsqueeze(raw, 0);
      auto s = squeeze_axis(u0, 2);
      return s->repeat({B, 1, 1});
    } else {
      // assume already [B,S,D]
      return raw;
    }
  } else if (raw->shape.size() == 4) {
    int b = raw->shape[0];
    int s = raw->shape[1];
    int one = raw->shape[2];
    int d = raw->shape[3];
    if (one == 1) {
      auto s3 = squeeze_axis(raw, 2);
      if (b == B)
        return s3;
      return s3->repeat({B, 1, 1});
    } else {
      throw std::runtime_error("unexpected rope cache shape");
    }
  }
  throw std::runtime_error("unsupported rope cache rank");
}

static std::string shape_str(const TensorPtr &t) {
  std::string s = "[";
  for (size_t i = 0; i < t->shape.size(); ++i) {
    s += std::to_string(t->shape[i]);
    if (i + 1 < t->shape.size())
      s += ",";
  }
  s += "]";
  return s;
}

static bool test_attention(const std::shared_ptr<Context> &ctx,
                           const fs::path &dir) {
  HostTensor Xh, Wqh, Wkh, Wvh, Woh, bqh, bkh, bvh, boh, pos_h, cos_h, sin_h,
      Yh;
  if (!read_tensor_bin(dir / "attn_X.tensor", Xh))
    return false;
  if (!read_tensor_bin(dir / "attn_Wq.tensor", Wqh))
    return false;
  if (!read_tensor_bin(dir / "attn_Wk.tensor", Wkh))
    return false;
  if (!read_tensor_bin(dir / "attn_Wv.tensor", Wvh))
    return false;
  if (!read_tensor_bin(dir / "attn_Wo.tensor", Woh))
    return false;
  if (!read_tensor_bin(dir / "attn_bq.tensor", bqh))
    return false;
  if (!read_tensor_bin(dir / "attn_bk.tensor", bkh))
    return false;
  if (!read_tensor_bin(dir / "attn_bv.tensor", bvh))
    return false;
  if (!read_tensor_bin(dir / "attn_bo.tensor", boh))
    return false;
  if (!read_tensor_bin(dir / "attn_pos_ids.tensor", pos_h))
    return false;
  if (!read_tensor_bin(dir / "attn_cos.tensor", cos_h))
    return false;
  if (!read_tensor_bin(dir / "attn_sin.tensor", sin_h))
    return false;
  if (!read_tensor_bin(dir / "attn_Y.tensor", Yh))
    return false;

  auto X = make_tensor(ctx, Xh.shape, Xh.data);
  auto Wq = make_tensor(ctx, Wqh.shape, Wqh.data);
  auto Wk = make_tensor(ctx, Wkh.shape, Wkh.data);
  auto Wv = make_tensor(ctx, Wvh.shape, Wvh.data);
  auto Wo = make_tensor(ctx, Woh.shape, Woh.data);
  auto bq = make_tensor(ctx, bqh.shape, bqh.data);
  auto bk = make_tensor(ctx, bkh.shape, bkh.data);
  auto bv = make_tensor(ctx, bvh.shape, bvh.data);
  auto bo = make_tensor(ctx, boh.shape, boh.data);
  auto Y_ref = make_tensor(ctx, Yh.shape, Yh.data);

  int B = X->shape[0];
  int S = X->shape[1];
  int Hsz = X->shape[2];
  // Infer head_dim D from RoPE cache, then derive H
  int D = 0;
  if ((int)cos_h.shape.size() == 3) {
    D = cos_h.shape[2];
  } else if ((int)cos_h.shape.size() == 4) {
    D = cos_h.shape[3];
  } else {
    throw std::runtime_error("attn_cos tensor rank must be 3 or 4");
  }
  if (D <= 0 || (Hsz % D) != 0)
    throw std::runtime_error("invalid head_dim inferred from RoPE cache");
  int H = Hsz / D;

  Qwen3VLTextConfig cfg;
  cfg.hidden_size = Hsz;
  cfg.num_attention_heads = H;
  cfg.num_key_value_heads = H;
  cfg.head_dim = D;
  cfg.max_position_embeddings = std::max(64, S + 1);
  cfg.attention_bias = true;
  cfg.rms_norm_eps = 1e-6;

  Qwen3Attention attn(ctx, cfg, /*layer_idx=*/0);
  // Copy weights/biases into attn
  for (size_t i = 0; i < attn.q_proj_weight->nelements(); ++i)
    attn.q_proj_weight->set_from_float_flat(i, Wq->get_as_float_flat(i));
  for (size_t i = 0; i < attn.k_proj_weight->nelements(); ++i)
    attn.k_proj_weight->set_from_float_flat(i, Wk->get_as_float_flat(i));
  for (size_t i = 0; i < attn.v_proj_weight->nelements(); ++i)
    attn.v_proj_weight->set_from_float_flat(i, Wv->get_as_float_flat(i));
  for (size_t i = 0; i < attn.o_proj_weight->nelements(); ++i)
    attn.o_proj_weight->set_from_float_flat(i, Wo->get_as_float_flat(i));

  for (size_t i = 0; i < attn.q_proj_bias->nelements(); ++i)
    attn.q_proj_bias->set_from_float_flat(i, bq->get_as_float_flat(i));
  for (size_t i = 0; i < attn.k_proj_bias->nelements(); ++i)
    attn.k_proj_bias->set_from_float_flat(i, bk->get_as_float_flat(i));
  for (size_t i = 0; i < attn.v_proj_bias->nelements(); ++i)
    attn.v_proj_bias->set_from_float_flat(i, bv->get_as_float_flat(i));
  for (size_t i = 0; i < attn.o_proj_bias->nelements(); ++i)
    attn.o_proj_bias->set_from_float_flat(i, bo->get_as_float_flat(i));

  auto cos_raw = make_tensor(ctx, cos_h.shape, cos_h.data);
  auto sin_raw = make_tensor(ctx, sin_h.shape, sin_h.data);
  auto cos_bsd = adapt_rope_cache_to_BSD(cos_raw, B);
  auto sin_bsd = adapt_rope_cache_to_BSD(sin_raw, B);

  std::cout << "[debug] cos_raw_shape=" << shape_str(cos_raw)
            << ", sin_raw_shape=" << shape_str(sin_raw) << std::endl;
  std::cout << "[debug] cos_bsd_shape=" << shape_str(cos_bsd)
            << ", sin_bsd_shape=" << shape_str(sin_bsd) << std::endl;

  TensorPtr Y;
  try {
    Y = attn.forward(X, cos_bsd, sin_bsd);
  } catch (const std::exception &e) {
    std::cerr << "[attention] forward exception: " << e.what() << std::endl;
    return false;
  }

  float mad = max_abs_diff(Y, Y_ref);
  std::cout << "[attention] max_abs_diff=" << mad << std::endl;
  return mad < 1e-4f;
}

int main(int argc, char **argv) {
  fs::path root = fs::path("tests/data");
  if (argc >= 2) {
    root = fs::path(argv[1]);
  }
  std::cout << "[ops_tests] data root=" << root << std::endl;
  auto ctx = std::make_shared<Context>();

  bool ok_linear = test_linear(ctx, root / "linear");
  bool ok_act = test_activations(ctx, root / "activations");
  bool ok_rms = test_rmsnorm(ctx, root / "rmsnorm");
  bool ok_attn = test_attention(ctx, root / "attention");

  std::cout << "[summary] linear="
            << ok_linear
            << ", activations=" << ok_act
            << ", rmsnorm=" << ok_rms
            << ", attention=" << ok_attn
            << std::endl;

  bool all_ok = ok_linear && ok_act && ok_rms && ok_attn;
  std::cout << (all_ok ? "PASS" : "FAIL") << std::endl;
  return all_ok ? 0 : 1;
}