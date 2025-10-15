// 简单入口程序，演示库初始化与版本信息
#include "include/common.h"
#include "include/ow_nn.h"
#include "src/context.hpp"
#include "src/tensor.hpp"
#include "src/cgraph.hpp"
#include "src/safetensors_loader.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <filesystem>

// --------- 轻量工具：权重名称匹配与摘要打印 ---------
static const char* dtype_to_cstr(ow::nn::DType dt);
static bool str_contains(const std::string &s, const std::string &sub) {
  return s.find(sub) != std::string::npos;
}

static void print_tensor_summary(const std::string &nm,
                                 ow::nn::SafetensorsLoader &loader,
                                 const std::shared_ptr<ow::nn::Context> &ctx,
                                 bool show_samples = true,
                                 size_t sample_n = 5) {
  const auto *entry = loader.get_entry(nm);
  auto T = loader.make_tensor(nm, ctx, false);
  size_t expect_bytes = entry ? entry->nbytes : 0;
  std::string orig = entry ? entry->st_dtype : std::string("?");
  std::cout << "  name= " << nm << ", dtype=" << dtype_to_cstr(T->dtype)
            << " (orig=" << orig << ")" << ", shape=[";
  for (size_t i = 0; i < T->shape.size(); ++i) {
    std::cout << T->shape[i];
    if (i + 1 < T->shape.size()) std::cout << ",";
  }
  std::cout << "] nbytes(expect/T)=" << expect_bytes << "/" << T->nbytes();
  if (show_samples) {
    size_t show = std::min<size_t>(sample_n, T->nelements());
    if (show > 0) {
      std::cout << " sample: ";
      for (size_t i = 0; i < show; ++i)
        std::cout << T->get_as_float_flat(i) << " ";
    }
  }
  std::cout << "\n";
}

struct ModuleInfo {
  std::string name;
  std::string type; // Embedding / Linear / LayerNorm / Other
};

static std::vector<ModuleInfo>
map_modules(const std::vector<std::string> &names) {
  std::vector<ModuleInfo> out;
  out.reserve(names.size());
  for (auto &nm : names) {
    std::string ty = "Other";
    if ((str_contains(nm, "embed") || str_contains(nm, "wte")) &&
        str_contains(nm, "weight")) {
      ty = "Embedding";
    } else if (str_contains(nm, "lm_head") && str_contains(nm, "weight")) {
      ty = "Linear";
    } else if ((str_contains(nm, "layer_norm") || str_contains(nm, "ln") ||
                str_contains(nm, "norm")) &&
               (str_contains(nm, "weight") || str_contains(nm, "bias"))) {
      ty = "LayerNorm";
    } else if ((str_contains(nm, "q_proj") || str_contains(nm, "k_proj") ||
                str_contains(nm, "v_proj") || str_contains(nm, "o_proj")) &&
               (str_contains(nm, "weight") || str_contains(nm, "bias"))) {
      ty = "Linear";
    }
    out.push_back(ModuleInfo{nm, ty});
  }
  return out;
}

static std::string find_first_matching(const std::vector<std::string> &names,
                                       const std::vector<std::string> &keys) {
  for (auto &nm : names) {
    for (auto &k : keys) {
      if (str_contains(nm, k)) return nm;
    }
  }
  return std::string();
}

// Embedding 前向：从权重 [V, D] 或 [D, V] 中按 token id 汇聚为 [1, D]
static ow::nn::TensorPtr forward_embedding_one(
    int token_id, const ow::nn::TensorPtr &W) {
  auto ctx = W->ctx.lock();
  if (!ctx) throw std::runtime_error("ctx expired");
  int V = W->shape[0];
  int D = (W->shape.size() >= 2) ? W->shape[1] : 1;
  // 期望权重是 [V, D]
  if (W->shape.size() != 2) {
    throw std::runtime_error("Embedding weight must be 2D");
  }
  if (token_id < 0 || token_id >= V) {
    throw std::runtime_error("token_id out of range for embedding");
  }
  auto H = ow::nn::Tensor::create(ctx, {1, D}, ow::nn::DType::FLOAT32);
  for (int j = 0; j < D; ++j) {
    size_t idx = (size_t)token_id * (size_t)D + (size_t)j;
    float v = W->get_as_float_flat(idx);
    H->set_from_float_flat((size_t)j, v);
  }
  return H;
}

// 线性前向：hidden [1, D] 与 W: [D, V] 或 [V, D]（自动处理转置）生成 logits [1, V]
static ow::nn::TensorPtr forward_linear_hidden_to_logits(
    const ow::nn::TensorPtr &H, const ow::nn::TensorPtr &W) {
  auto ctx = H->ctx.lock();
  if (!ctx) throw std::runtime_error("ctx expired");
  if (H->shape.size() != 2 || H->shape[0] != 1)
    throw std::runtime_error("hidden must be [1, D]");
  int D = H->shape[1];
  if (W->shape.size() != 2)
    throw std::runtime_error("linear weight must be 2D");
  int r = W->shape[0];
  int c = W->shape[1];
  ow::nn::TensorPtr logits;
  if (r == D) {
    int V = c;
    auto HH = ow::nn::Tensor::create(ctx, {1, D}, ow::nn::DType::FLOAT32);
    // 拷贝 H 到 HH（保证内存连续二维视图）
    for (int i = 0; i < D; ++i)
      HH->set_from_float_flat((size_t)i, H->get_as_float_flat((size_t)i));
    logits = ow::nn::Tensor::matmul_blocked_mt(HH, W, 64, 64, 64, 1);
    (void)V;
  } else if (c == D) {
    int V = r; // W 是 [V, D]，等价于 H @ W^T
    logits = ow::nn::Tensor::create(ctx, {1, V}, ow::nn::DType::FLOAT32);
    for (int j = 0; j < V; ++j) {
      double acc = 0.0;
      for (int i = 0; i < D; ++i) {
        float h = H->get_as_float_flat((size_t)i);
        float w = W->get_as_float_flat((size_t)j * (size_t)D + (size_t)i);
        acc += (double)h * (double)w;
      }
      logits->set_from_float_flat((size_t)j, (float)acc);
    }
  } else {
    throw std::runtime_error("linear weight dims not compatible with hidden");
  }
  return logits;
}

void fill_tensor_from_vector(const ow::nn::TensorPtr &T,
                             const std::vector<float> &vals) {
  if (T->nelements() != vals.size()) {
    std::runtime_error("fill size");
  }
  for (size_t i = 0; i < vals.size(); ++i)
    T->set_from_float_flat(i, vals[i]);
}

static const char* dtype_to_cstr(ow::nn::DType dt){
  switch(dt){
    case ow::nn::DType::FLOAT32: return "FLOAT32";
    case ow::nn::DType::INT32: return "INT32";
    case ow::nn::DType::FP16: return "FP16";
    case ow::nn::DType::INT8: return "INT8";
    case ow::nn::DType::Q4_0: return "Q4_0";
    default: return "UNKNOWN";
  }
}

// ---------------- Image IO helpers (PPM P6) and simple nearest resize
static bool load_ppm_p6_rgb(const std::string &path,
                            std::vector<float> &rgb,
                            int &H, int &W) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;
  std::string magic;
  ifs >> magic;
  if (magic != std::string("P6")) return false;
  auto skip_comments = [&]() {
    while (ifs.peek() == '#') {
      std::string line; std::getline(ifs, line);
    }
  };
  skip_comments();
  ifs >> W; skip_comments();
  ifs >> H; skip_comments();
  int maxv = 255; ifs >> maxv;
  if (maxv <= 0) return false;
  ifs.get(); // consume single whitespace after header
  std::vector<uint8_t> buf((size_t)H * (size_t)W * 3);
  ifs.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
  if ((size_t)ifs.gcount() != buf.size()) return false;
  rgb.resize(buf.size());
  for (size_t i = 0; i < buf.size(); ++i) {
    float v01 = float(buf[i]) / float(maxv);
    rgb[i] = v01 * 2.0f - 1.0f; // normalize to [-1,1]
  }
  return true;
}

static std::vector<float> resize_nearest_rgb(const std::vector<float> &src,
                                             int H, int W,
                                             int H2, int W2) {
  std::vector<float> dst((size_t)H2 * (size_t)W2 * 3);
  float ry = float(H) / float(H2);
  float rx = float(W) / float(W2);
  for (int y2 = 0; y2 < H2; ++y2) {
    int y = std::min(H - 1, int(y2 * ry));
    for (int x2 = 0; x2 < W2; ++x2) {
      int x = std::min(W - 1, int(x2 * rx));
      for (int c = 0; c < 3; ++c) {
        size_t si = ((size_t)y * (size_t)W + (size_t)x) * 3 + (size_t)c;
        size_t di = ((size_t)y2 * (size_t)W2 + (size_t)x2) * 3 + (size_t)c;
        dst[di] = src[si];
      }
    }
  }
  return dst;
}

static ow::nn::TensorPtr make_image_stack_tensor(
    const std::shared_ptr<ow::nn::Context> &ctx,
    const std::vector<std::string> &paths,
    int targetH = 224, int targetW = 224) {
  int D = int(paths.size()) * 2; // two depth slices per image
  auto X = ow::nn::Tensor::create(ctx, {3, D, targetH, targetW}, ow::nn::DType::FLOAT32);
  for (size_t i = 0; i < paths.size(); ++i) {
    std::vector<float> rgb; int H = 0, W = 0;
    if (!load_ppm_p6_rgb(paths[i], rgb, H, W)) {
      // fill with zeros on failure
      rgb.assign((size_t)targetH * (size_t)targetW * 3, 0.0f);
    } else {
      if (H != targetH || W != targetW) rgb = resize_nearest_rgb(rgb, H, W, targetH, targetW);
    }
    // write into X for depth slices 2*i and 2*i+1
    for (int drep = 0; drep < 2; ++drep) {
      int d = int(i) * 2 + drep;
      for (int y = 0; y < targetH; ++y) {
        for (int x = 0; x < targetW; ++x) {
          for (int c = 0; c < 3; ++c) {
            float v = rgb[((size_t)y * (size_t)targetW + (size_t)x) * 3 + (size_t)c];
            size_t idx = (((size_t)c * (size_t)X->shape[1] + (size_t)d) * (size_t)targetH + (size_t)y) * (size_t)targetW + (size_t)x;
            X->set_from_float_flat(idx, v);
          }
        }
      }
    }
  }
  return X;
}

// Find weight names by shape heuristics
static std::string find_conv3d_weight_3to1280_k2_14_14(
    ow::nn::SafetensorsLoader &loader,
    const std::vector<std::string> &names) {
  for (auto &nm : names) {
    auto e = loader.get_entry(nm);
    if (!e) continue;
    const auto &sh = e->shape;
    if (sh.size() == 5 && sh[0] == 1280 && sh[1] == 3 && sh[2] == 2 && sh[3] == 14 && sh[4] == 14)
      return nm;
  }
  return "";
}

static std::string find_linear_1280_to_embed(
    ow::nn::SafetensorsLoader &loader,
    const std::vector<std::string> &names,
    int embed_dim,
    bool &is_transposed_out) {
  // prefer W with shape [1280, embed_dim]
  for (auto &nm : names) {
    auto e = loader.get_entry(nm);
    if (!e) continue;
    const auto &sh = e->shape;
    if (sh.size() == 2 && sh[0] == 1280 && sh[1] == embed_dim) {
      is_transposed_out = false; return nm;
    }
  }
  // accept transposed shape [embed_dim, 1280]
  for (auto &nm : names) {
    auto e = loader.get_entry(nm);
    if (!e) continue;
    const auto &sh = e->shape;
    if (sh.size() == 2 && sh[0] == embed_dim && sh[1] == 1280) {
      is_transposed_out = true; return nm;
    }
  }
  return "";
}

static ow::nn::TensorPtr transpose_copy_2d(const ow::nn::TensorPtr &B) {
  auto ctx = B->ctx.lock(); if (!ctx) throw std::runtime_error("ctx expired");
  if (B->shape.size() != 2) throw std::runtime_error("transpose_copy_2d: need 2D");
  int r = B->shape[0], c = B->shape[1];
  auto BT = ow::nn::Tensor::create(ctx, {c, r}, ow::nn::DType::FLOAT32);
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      float v = B->get_as_float_flat((size_t)i * (size_t)c + (size_t)j);
      BT->set_from_float_flat((size_t)j * (size_t)r + (size_t)i, v);
    }
  }
  return BT;
}

static ow::nn::TensorPtr project_tokens_1280_to_embed(
    const ow::nn::TensorPtr &tokens, const ow::nn::TensorPtr &W,
    bool W_is_transposed) {
  // tokens: [Ntoks, 1280]
  if (tokens->shape.size() != 2 || tokens->shape[1] != 1280)
    throw std::runtime_error("project_tokens: tokens must be [Ntoks,1280]");
  if (W->shape.size() != 2) throw std::runtime_error("project_tokens: W must be 2D");
  if (W_is_transposed) {
    auto WT = transpose_copy_2d(W); // to [1280, embed]
    return ow::nn::Tensor::matmul_blocked_mt(tokens, WT);
  } else {
    return ow::nn::Tensor::matmul_blocked_mt(tokens, W);
  }
}

int main(int argc, char **argv) {
  std::cout << "ow_nn main executable running" << std::endl;

  // Use a larger arena to avoid pointer invalidation due to vector reallocation
  auto ctx = std::make_shared<ow::nn::Context>(256 * 1024 * 1024);
  // create tensors for matmul test: A (256*512) B (512*256) --sizes chosen to
  // exercise blocking
  int M = 256, K = 512, N = 256;
  auto A = ow::nn::Tensor::create(ctx, {M, K}, ow::nn::DType::FLOAT32);
  auto B = ow::nn::Tensor::create(ctx, {K, N}, ow::nn::DType::FP16);
  // fill A and B with small values
  std::vector<float> avals(M * K), bvals(K * N);
  for (int i = 0; i < M * K; ++i)
    avals[i] = (i % 7 - 3) * 0.01f + 0.5f;
  for (int i = 0; i < K * N; ++i)
    bvals[i] = (i % 11 - 5) * 0.02f + 1.0f;
  fill_tensor_from_vector(A, avals);
  fill_tensor_from_vector(B, bvals);
  // run blocked multithreaded matmul
  auto R = ow::nn::Tensor::matmul_blocked_mt(
      A, B, 64, 64, 64, std::thread::hardware_concurrency());
  std::cout << "Matmul done. R[0..7]: ";
  for (int i = 0; i < 8; ++i) {
    std::cout << R->get_as_float_flat(i) << " ";
  }
  std::cout << " \n";

  // quick numeric check vs naive reference
  auto Rref = ow::nn::Tensor::create(ctx, {M, N}, ow::nn::DType::FLOAT32);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int p = 0; p < K; ++p) {
        acc += A->get_as_float_flat((size_t)i * K + p) *
               B->get_as_float_flat((size_t)p * N + j);
      }
      Rref->set_from_float_flat((size_t)i * N + j, acc);
    }
  }
  double max_abs_err = 0.0;
  for (size_t idx = 0; idx < R->nelements(); ++idx) {
    double e = std::abs(R->get_as_float_flat(idx) - Rref->get_as_float_flat(idx));
    if (e > max_abs_err) max_abs_err = e;
  }
  std::cout << "Matmul max_abs_err vs ref: " << max_abs_err << "\n";

  // benchmarking optimized matmul (non-interactive)
  auto bench = [](int M, int K, int N, int repeats) {
    auto ctxb = std::make_shared<ow::nn::Context>(64 * 1024 * 1024);
    auto A = ow::nn::Tensor::create(ctxb, {M, K}, ow::nn::DType::FLOAT32);
    auto B = ow::nn::Tensor::create(ctxb, {K, N}, ow::nn::DType::FP16);
    std::vector<float> avals((size_t)M * K), bvals((size_t)K * N);
    for (size_t i = 0; i < avals.size(); ++i)
      avals[i] = float((i % 7) - 3) * 0.01f + 0.5f;
    for (size_t i = 0; i < bvals.size(); ++i)
      bvals[i] = float((i % 11) - 5) * 0.02f + 1.0f;
    fill_tensor_from_vector(A, avals);
    fill_tensor_from_vector(B, bvals);

    double total_ms = 0.0;
    for (int r = 0; r < repeats; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      auto Ropt = ow::nn::Tensor::matmul_blocked_mt(
          A, B, 64, 64, 128, std::thread::hardware_concurrency());
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> dt = t1 - t0;
      total_ms += dt.count();
      (void)Ropt;
    }
    std::cout << "Benchmark M=" << M << " K=" << K << " N=" << N
              << ": avg " << (total_ms / repeats) << " ms" << std::endl;
  };
  // run a couple of sizes; keep repeats small to avoid long runtime
  bench(512, 512, 512, 3);
  bench(1024, 1024, 1024, 2);
  // quantize example: int8
  auto C_int8 = ow::nn::Tensor::create(ctx, {M, K}, ow::nn::DType::INT8);
  ow::nn::Tensor::quantize_int8_from_floats(C_int8, avals);
  std::cout << "C_int8 sample: ";
  for (int i = 0; i < 8; ++i) {
    std::cout << C_int8->get_as_float_flat(i) << " ";
  }
  std::cout << "\n";

  // q4_0 example
  auto Q = ow::nn::Tensor::create(ctx, {8}, ow::nn::DType::Q4_0);
  std::vector<float> qsrc = {0.5f, 1.0f, -0.5f, 2.0f, -1.5f, 3.0f, -2.0f, 0.1f};
  ow::nn::Tensor::quantize_q4_0_from_floats(Q, qsrc);
  Q->print_f("Q4_0 dequantized");

  // small compute graph demo: C = A_row0 @ B_col0 (dot) then add scalar
  ow::nn::ComputeGraph graph(ctx);
  graph.add_input("A", A);
  graph.add_input("B", B);

  ow::nn::OpNode n1;
  n1.name = "matmul1";
  n1.inputs = {"A", "B"};
  n1.output = "R";
  n1.fn = [ctx](const std::vector<ow::nn::TensorPtr> &ins,
                const std::unordered_map<std::string, std::string> & /*attrs*/) {
    auto A = ins[0];
    auto B = ins[1];
    auto R = ow::nn::Tensor::matmul_blocked_mt(
        A, B, 64, 64, 64,
        std::thread::hardware_concurrency()); /* register into map by capturing
                                                 graph? we'll do via global
                                                 trick below */
    return R;
  };
  // Because the compute-graph lambda cannot directly insert into graph.tensors
  // (capturing this),
  // for this minimal demo we will instead run the matmul directly here and
  // manually continue.

  // Simpler: demonstrate graph with a trivial add node
  // create two small tensors and run add via graph

  auto X = ow::nn::Tensor::create(ctx, {4}, ow::nn::DType::FLOAT32);
  fill_tensor_from_vector(X, {1, 2, 3, 4});
  auto Y = ow::nn::Tensor::create(ctx, {4}, ow::nn::DType::FLOAT32);
  fill_tensor_from_vector(Y, {0.5f, 0.5f, 0.5f, 0.5f});
  graph.add_input("X", X);
  graph.add_input("Y", Y);
  ow::nn::OpNode addn;
  addn.name = "add1";
  addn.inputs = {"X", "Y"};
  std::string outZ = "Z";
  addn.output = outZ;
  addn.fn = [&graph, outZ](const std::vector<ow::nn::TensorPtr> &ins,
                           const std::unordered_map<std::string, std::string> & /*attrs*/) {
    auto A = ins[0];
    auto B = ins[1];
    auto ctx = A->ctx.lock();
    auto R = ow::nn::Tensor::create(ctx, A->shape, ow::nn::DType::FLOAT32);
    // element-wise add
    for (size_t i = 0; i < A->nelements(); ++i) {
      float v = A->get_as_float_flat(i) + B->get_as_float_flat(i);
      R->set_from_float_flat(i, v);
    }
    return R;
  };
  graph.add_node(addn);
  graph.run();
  if (graph.tensors.count(outZ)) {
    auto Z = graph.tensors[outZ];
    std::cout << "Graph add Z[0..3]: ";
    for (int i = 0; i < 4; ++i) {
      std::cout << Z->get_as_float_flat(i) << " ";
    }
    std::cout << "\n";
  }

  // ---- HuggingFace safetensors loader demo ----
  try {
    std::string model_dir = "model";
    std::string abs_model_dir = "d:/workspace/cpp_projects/ow_nn/model";
    if (std::filesystem::exists(abs_model_dir)) model_dir = abs_model_dir;
    else if (!std::filesystem::exists(model_dir)) {
      std::cout << "Model directory not found, skip safetensors demo." << std::endl;
      goto skip_st_demo;
    }
    ow::nn::SafetensorsLoader loader;
    loader.load_dir(model_dir);
    auto names = loader.names_supported();
    std::cout << "Loaded safetensors dir: " << model_dir << ", tensors: " << names.size() << std::endl;
    int print_count = 0;
    for (auto &nm : names) {
      try {
        print_tensor_summary(nm, loader, ctx, true, 5);
        if (++print_count >= 10) break;
      } catch (const std::exception &et) {
        std::cout << "  name= " << nm << " [skip unsupported dtype] reason: " << et.what() << "\n";
        continue;
      }
    }

    // 建立命名到模块映射，并统计关键模块
    auto mods = map_modules(names);
    size_t n_embed = 0, n_linear = 0, n_ln = 0;
    for (auto &m : mods) {
      if (m.type == "Embedding") ++n_embed;
      else if (m.type == "Linear") ++n_linear;
      else if (m.type == "LayerNorm") ++n_ln;
    }
    std::cout << "Module map: Embedding=" << n_embed << ", Linear=" << n_linear
              << ", LayerNorm=" << n_ln << ", Other="
              << (mods.size() - n_embed - n_linear - n_ln) << "\n";

    // 选择并验证特定权重
    std::string nm_embed = find_first_matching(
        names, {"token_embed.weight", "embed_tokens.weight", "wte.weight",
                "model.embed_tokens.weight", "embed.weight", "embedding.weight"});
    std::string nm_lm = find_first_matching(names, {"lm_head.weight", "lm_head"});
    std::string nm_q = find_first_matching(names, {"q_proj.weight", "self_attn.q_proj.weight"});
    std::string nm_k = find_first_matching(names, {"k_proj.weight", "self_attn.k_proj.weight"});
    std::string nm_v = find_first_matching(names, {"v_proj.weight", "self_attn.v_proj.weight"});
    std::string nm_o = find_first_matching(names, {"o_proj.weight", "self_attn.o_proj.weight"});

    if (!nm_embed.empty()) {
      std::cout << "[Target] token_embed.weight summary:\n";
      print_tensor_summary(nm_embed, loader, ctx, true, 8);
    } else {
      std::cout << "[Target] token_embed.weight not found by heuristics." << std::endl;
    }
    if (!nm_lm.empty()) {
      std::cout << "[Target] lm_head.weight summary:\n";
      print_tensor_summary(nm_lm, loader, ctx, true, 8);
    } else {
      std::cout << "[Target] lm_head.weight not found by heuristics." << std::endl;
    }
    if (!nm_q.empty()) { std::cout << "[Target] q_proj.weight\n"; print_tensor_summary(nm_q, loader, ctx, false); }
    if (!nm_k.empty()) { std::cout << "[Target] k_proj.weight\n"; print_tensor_summary(nm_k, loader, ctx, false); }
    if (!nm_v.empty()) { std::cout << "[Target] v_proj.weight\n"; print_tensor_summary(nm_v, loader, ctx, false); }
    if (!nm_o.empty()) { std::cout << "[Target] o_proj.weight\n"; print_tensor_summary(nm_o, loader, ctx, false); }

    // 最小前向图（ggml-like）：token -> embedding -> logits
    if (!nm_embed.empty() && !nm_lm.empty()) {
      auto Wembed = loader.make_tensor(nm_embed, ctx, false);
      auto Wlm = loader.make_tensor(nm_lm, ctx, false);

      ow::nn::ComputeGraph mgraph(ctx);
      // 输入：token_ids（示例用单个 token 42）与权重
      auto token_ids = ow::nn::Tensor::create(ctx, {1}, ow::nn::DType::FLOAT32);
      token_ids->set_from_float_flat(0, 42.0f);
      mgraph.add_input("token_ids", token_ids);
      mgraph.add_input("embed_weight", Wembed);
      mgraph.add_input("lm_head_weight", Wlm);

      // Embedding 节点
      ow::nn::OpNode n_embed;
      n_embed.name = "embedding";
      n_embed.inputs = {"token_ids", "embed_weight"};
      n_embed.output = "hidden";
      n_embed.fn = [&mgraph](const std::vector<ow::nn::TensorPtr> &ins,
                             const std::unordered_map<std::string, std::string> & /*attrs*/) {
        int token = (int)ins[0]->get_as_float_flat(0);
        auto H = forward_embedding_one(token, ins[1]);
        return H;
      };
      mgraph.add_node(n_embed);

      // Linear 节点（到 logits）
      ow::nn::OpNode n_linear;
      n_linear.name = "lm_head_linear";
      n_linear.inputs = {"hidden", "lm_head_weight"};
      n_linear.output = "logits";
      n_linear.fn = [&mgraph](const std::vector<ow::nn::TensorPtr> &ins,
                              const std::unordered_map<std::string, std::string> & /*attrs*/) {
        auto logits = forward_linear_hidden_to_logits(ins[0], ins[1]);
        return logits;
      };
      mgraph.add_node(n_linear);

      mgraph.run();
      if (mgraph.tensors.count("logits")) {
        auto L = mgraph.tensors["logits"];
        std::cout << "Forward logits sample[0..9]: ";
        for (int i = 0; i < std::min<int>(10, L->shape[1]); ++i) {
          std::cout << L->get_as_float_flat(i) << " ";
        }
        std::cout << "\n";
      }

      // ---- Visual Prefix Graph (integrated, using real weights when available)
      int embed_dim = 0;
      if (!nm_embed.empty()) {
        if (auto e = loader.get_entry(nm_embed)) {
          if (e->shape.size() == 2) embed_dim = e->shape[1];
        }
      }
      if (embed_dim <= 0) {
        embed_dim = 4096; // fallback typical dim
        std::cout << "[Vision] embed dim fallback to " << embed_dim << "\n";
      } else {
        std::cout << "[Vision] text embed dim detected: " << embed_dim << "\n";
      }

      // Collect PPM images from common folders
      std::vector<std::string> ppm_paths;
      auto try_collect_dir = [&](const std::filesystem::path &dir) {
        if (!std::filesystem::exists(dir)) return;
        for (auto &de : std::filesystem::directory_iterator(dir)) {
          if (!de.is_regular_file()) continue;
          auto p = de.path();
          auto ext = p.extension().string();
          for (auto &ch : ext) ch = (char)std::tolower(ch);
          if (ext == ".ppm") {
            ppm_paths.push_back(p.string());
            if (ppm_paths.size() >= 4) break; // cap
          }
        }
      };
      try_collect_dir(std::filesystem::path("assert/images"));
      try_collect_dir(std::filesystem::path("images"));

      ow::nn::TensorPtr Ximg;
      if (!ppm_paths.empty()) {
        Ximg = make_image_stack_tensor(ctx, ppm_paths, 224, 224);
      } else {
        // Fallback synthetic image: {C=3, D=2, H=224, W=224}
        std::cout << "[Vision] No .ppm found; using synthetic image" << std::endl;
        Ximg = ow::nn::Tensor::create(ctx, {3, 2, 224, 224}, ow::nn::DType::FLOAT32);
        for (size_t i = 0; i < Ximg->nelements(); ++i) {
          int kk = int(i % 101) - 50;
          float v = float(kk) * 0.0005f;
          Ximg->set_from_float_flat(i, v);
        }
      }

      // Detect Conv3d and Linear(1280->embed) weights by shape
      std::string nm_conv = find_conv3d_weight_3to1280_k2_14_14(loader, names);
      bool proj_transposed = false;
      std::string nm_proj = find_linear_1280_to_embed(loader, names, embed_dim, proj_transposed);

      ow::nn::TensorPtr Wconv;
      if (!nm_conv.empty()) {
        Wconv = loader.make_tensor(nm_conv, ctx, false);
        std::cout << "[Vision] conv3d weight: " << nm_conv << " dtype=" << dtype_to_cstr(Wconv->dtype) << " shape="
                  << Wconv->shape[0] << "," << Wconv->shape[1] << ","
                  << Wconv->shape[2] << "," << Wconv->shape[3] << ","
                  << Wconv->shape[4] << "\n";
      } else {
        std::cout << "[Vision] conv3d weight not found; using small random init" << std::endl;
        Wconv = ow::nn::Tensor::create(ctx, {1280, 3, 2, 14, 14}, ow::nn::DType::FLOAT32);
        for (size_t i = 0; i < Wconv->nelements(); ++i) {
          int kk = int(i % 37) - 18;
          float v = float(kk) * 0.0001f;
          Wconv->set_from_float_flat(i, v);
        }
        std::cout << "[Vision] Wconv dtype=" << dtype_to_cstr(Wconv->dtype)
                  << " nelements=" << Wconv->nelements()
                  << " nbytes=" << Wconv->nbytes() << std::endl;
        std::cout << "[Vision] Wconv sample[0..9]: ";
        for (int i = 0; i < 10; ++i) std::cout << Wconv->get_as_float_flat(i) << " ";
        std::cout << "\n";
        std::cout << "[Vision] Wconv bytes[0..9]: ";
        for (int i = 0; i < 10; ++i) {
          float fv = Wconv->get_as_float_flat(i);
          uint32_t u; std::memcpy(&u, &fv, 4);
          std::cout << std::hex << std::showbase << u << std::dec << " ";
        }
        std::cout << "\n";
        // overwrite first 10 values with known small constants
        for (int i = 0; i < 10; ++i) {
          Wconv->set_from_float_flat(i, (float)(i - 5) * 0.001f);
        }
        std::cout << "[Vision] Wconv sample fixed[0..9]: ";
        for (int i = 0; i < 10; ++i) std::cout << Wconv->get_as_float_flat(i) << " ";
        std::cout << "\n";
      }

      ow::nn::TensorPtr Wproj;
      if (!nm_proj.empty()) {
        Wproj = loader.make_tensor(nm_proj, ctx, false);
        std::cout << "[Vision] proj weight: " << nm_proj << " dtype=" << dtype_to_cstr(Wproj->dtype) << " shape="
                  << Wproj->shape[0] << "," << Wproj->shape[1]
                  << (proj_transposed ? " (transposed)" : "") << "\n";
      } else {
        std::cout << "[Vision] 1280->embed weight not found; using small random init" << std::endl;
        Wproj = ow::nn::Tensor::create(ctx, {1280, embed_dim}, ow::nn::DType::FLOAT32);
        for (size_t i = 0; i < Wproj->nelements(); ++i) {
          int kk = int(i % 41) - 20;
          float v = float(kk) * 0.0002f;
          Wproj->set_from_float_flat(i, v);
        }
        proj_transposed = false;
        std::cout << "[Vision] Wproj dtype=" << dtype_to_cstr(Wproj->dtype)
                  << " nelements=" << Wproj->nelements()
                  << " nbytes=" << Wproj->nbytes() << std::endl;
        std::cout << "[Vision] Wproj sample[0..9]: ";
        for (int i = 0; i < 10; ++i) std::cout << Wproj->get_as_float_flat(i) << " ";
        std::cout << "\n";
        std::cout << "[Vision] Wproj bytes[0..9]: ";
        for (int i = 0; i < 10; ++i) {
          float fv = Wproj->get_as_float_flat(i);
          uint32_t u; std::memcpy(&u, &fv, 4);
          std::cout << std::hex << std::showbase << u << std::dec << " ";
        }
        std::cout << "\n";
        for (int i = 0; i < 10; ++i) {
          Wproj->set_from_float_flat(i, (float)(i - 5) * 0.001f);
        }
        std::cout << "[Vision] Wproj sample fixed[0..9]: ";
        for (int i = 0; i < 10; ++i) std::cout << Wproj->get_as_float_flat(i) << " ";
        std::cout << "\n";
      }

      ow::nn::ComputeGraph vgraph(ctx);
      vgraph.add_input("image", Ximg);
      vgraph.add_input("convW", Wconv);
      vgraph.add_input("projW", Wproj);

      // Conv3d node
      ow::nn::OpNode nconv;
      nconv.name = "conv3d_patch";
      nconv.inputs = {"image", "convW"};
      nconv.output = "Yconv";
      nconv.fn = [](const std::vector<ow::nn::TensorPtr> &ins,
                    const std::unordered_map<std::string, std::string> & /*attrs*/) {
        return ow::nn::Tensor::conv3d_k21414_s21414(ins[0], ins[1]);
      };
      vgraph.add_node(nconv);

      // Flatten tokens node
      ow::nn::OpNode nft;
      nft.name = "flatten_tokens";
      nft.inputs = {"Yconv"};
      nft.output = "visual_tokens1280";
      nft.fn = [](const std::vector<ow::nn::TensorPtr> &ins,
                  const std::unordered_map<std::string, std::string> & /*attrs*/) {
        return ow::nn::Tensor::flatten_conv3d_tokens(ins[0]);
      };
      vgraph.add_node(nft);

      // Vision projection node
      ow::nn::OpNode nproj;
      nproj.name = "vision_proj";
      nproj.inputs = {"visual_tokens1280", "projW"};
      nproj.output = "visual_tokens";
      nproj.fn = [proj_transposed](const std::vector<ow::nn::TensorPtr> &ins,
                                  const std::unordered_map<std::string, std::string> & /*attrs*/) {
        return project_tokens_1280_to_embed(ins[0], ins[1], proj_transposed);
      };
      vgraph.add_node(nproj);

      vgraph.run();
      if (vgraph.tensors.count("Yconv")) {
        auto YC = vgraph.tensors["Yconv"];
        std::cout << "[Vision] Yconv sample[0..9]: ";
        for (int i = 0; i < std::min<int>(10, YC->nelements()); ++i) {
          std::cout << YC->get_as_float_flat(i) << " ";
        }
        std::cout << "\n";
      }
      if (vgraph.tensors.count("visual_tokens1280")) {
        auto VT1280 = vgraph.tensors["visual_tokens1280"];
        std::cout << "[Vision] tokens1280 shape=[" << VT1280->shape[0] << "," << VT1280->shape[1] << "] sample[0..9]: ";
        for (int i = 0; i < std::min<int>(10, VT1280->shape[1]); ++i) {
          std::cout << VT1280->get_as_float_flat(i) << " ";
        }
        std::cout << "\n";
      }
      if (vgraph.tensors.count("visual_tokens")) {
        auto VT = vgraph.tensors["visual_tokens"];
        std::cout << "[Vision] visual tokens shape=[" << VT->shape[0] << "," << VT->shape[1] << "] sample[0..9]: ";
        for (int i = 0; i < std::min<int>(10, VT->shape[1]); ++i) {
          std::cout << VT->get_as_float_flat(i) << " ";
        }
        std::cout << "\n";
      }
    } else {
      std::cout << "Skip minimal forward graph due to missing embed/lm_head." << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Safetensors demo error: " << e.what() << std::endl;
  }
skip_st_demo:


  std::string vocab_path = "assert/vocab.json";
  std::string merges_path = "assert/merges.txt";
  if (argc >= 3) {
    vocab_path = argv[1];
    merges_path = argv[2];
  } else {
    std::cout << "Using default assets: " << vocab_path << ", " << merges_path
              << std::endl;
    std::cout << "Usage: " << argv[0] << " <vocab.json> <merges.txt>"
              << std::endl;
  }

  try {
    ow::nn::Tokenizer tokenizer(vocab_path, merges_path);
    std::string input;
    std::cout << "Enter a sentence: " << std::flush;
    std::getline(std::cin, input);
    auto tokens = tokenizer.encode(input);
    std::cout << "Encoded IDs: ";
    for (int t : tokens)
      std::cout << t << " ";
    std::cout << std::endl;

    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded text: " << decoded << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}