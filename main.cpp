// 简单入口程序，演示库初始化与版本信息
#include "include/common.h"
#include "include/ow_nn.h"
#include "src/context.hpp"
#include "src/tensor.hpp"
#include "src/cgraph.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

void fill_tensor_from_vector(const ow::nn::TensorPtr &T,
                             const std::vector<float> &vals) {
  if (T->nelements() != vals.size()) {
    std::runtime_error("fill size");
  }
  for (size_t i = 0; i < vals.size(); ++i)
    T->set_from_float_flat(i, vals[i]);
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