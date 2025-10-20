#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdint.h>
#include <vector>

// simple single sequence output sampler without beam search
struct sampler {
  // runtime-configurable generation settings
  uint32_t eos_token_id = 0; // 0 means unknown/unused
  uint32_t vocab_size = 0;   // must be set from model weights
  float temperature = 1.0f;
  uint32_t top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f; // 1.0 disables
  std::vector<uint32_t> last_token_ids;
  bool do_sample = true;
  bool apply_softmax = true;
  size_t last_max_keep = 128; // keep at most N recent ids for repetition penalty

  void sample(float *logits, std::vector<uint32_t> &output_tokens);
  void softmax(float *logits, std::vector<std::vector<size_t>> picks,
               std::vector<uint32_t> max_indices);
  void max(float *logits, std::vector<uint32_t> &output_tokens);
  void topk(float *probs, std::vector<uint32_t>& out_indices);
  void topp(float *probs, std::vector<uint32_t>& out_indices);
  void reset(size_t keep_last_n = 128);
};