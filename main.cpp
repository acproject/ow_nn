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

int main(int argc, char **argv) {
  std::cout << "ow_nn main executable running" << std::endl;

  auto ctx = std::make_shared<ow::nn::Context>(4 * 1024 * 1024);
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
                const std::vector<ow::nn::TensorPtr> &outs) {
    auto A = ins[0];
    auto B = ins[1];
    auto R = ow::nn::Tensor::matmul_blocked_mt(
        A, B, 64, 64, 64,
        std::thread::hardware_concurrency()); /* register into map by capturing
                                                 graph? we'll do via global
                                                 trick below */
    (void)R;
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
                           const std::vector<ow::nn::TensorPtr> & /*outs*/) {
    auto A = ins[0];
    auto B = ins[1];
    auto ctx = A->ctx.lock();
    auto R = ow::nn::Tensor::create(ctx, A->shape, ow::nn::DType::FLOAT32);
    // element-wise add
    for (size_t i = 0; i < A->nelements(); ++i) {
      float v = A->get_as_float_flat(i) + B->get_as_float_flat(i);
      R->set_from_float_flat(i, v);
    }
    // register output tensor in graph
    graph.tensors[outZ] = R;
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
    auto names = loader.names();
    std::cout << "Loaded safetensors dir: " << model_dir << ", tensors: " << names.size() << std::endl;
    int print_count = 0;
    for (auto &nm : names) {
      auto T = loader.make_tensor(nm, ctx, false); // zero-copy view
      std::cout << "  name= " << nm << ", dtype=" << dtype_to_cstr(T->dtype) << ", shape=[";
      for (size_t i = 0; i < T->shape.size(); ++i) {
        std::cout << T->shape[i];
        if (i + 1 < T->shape.size()) std::cout << ",";
      }
      std::cout << "]";
      // print first few values if FLOAT32/FP16/INT32/INT8
      size_t show = std::min<size_t>(5, T->nelements());
      if (show > 0) {
        std::cout << " sample: ";
        for (size_t i = 0; i < show; ++i) std::cout << T->get_as_float_flat(i) << " ";
      }
      std::cout << "\n";
      if (++print_count >= 10) break;
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