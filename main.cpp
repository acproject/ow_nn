#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <queue>
#include <functional>
#include <cctype>

#include "include/ow_nn.h"
#include "include/transformer.h"
#include "src/safetensors_loader.hpp"
#include "src/context.hpp"
#include "src/tensor.hpp"
#include "include/tokenizer.h"

// -----------------------------------------------------------------------------
// Helpers (phase 1)
struct ModuleInfo {
  std::string name;
  std::string type; // Embedding / Linear / LayerNorm / Other
};

static bool str_contains(const std::string &s, const std::string &sub) {
  return s.find(sub) != std::string::npos;
}

static const char* dtype_to_cstr(ow::nn::DType dt) {
  switch (dt) {
    case ow::nn::DType::FLOAT32: return "float32";
    case ow::nn::DType::INT32:   return "int32";
    case ow::nn::DType::FP16:    return "fp16";
    case ow::nn::DType::BF16:    return "bf16";
    case ow::nn::DType::INT8:    return "int8";
    case ow::nn::DType::Q4_0:    return "q4_0";
    default: return "unknown";
  }
}

static std::vector<ModuleInfo> map_modules(const std::vector<std::string> &names) {
  std::vector<ModuleInfo> mods;
  mods.reserve(names.size());
  for (auto &nm : names) {
    ModuleInfo m; m.name = nm; m.type = "Other";
    if (str_contains(nm, "embed") || str_contains(nm, "Embedding") || str_contains(nm, "wte")) {
      m.type = "Embedding";
    } else if (str_contains(nm, "lm_head") || str_contains(nm, "Linear") || str_contains(nm, ".weight")) {
      m.type = "Linear";
    } else if (str_contains(nm, "layer_norm") || str_contains(nm, "LayerNorm") || str_contains(nm, "ln")) {
      m.type = "LayerNorm";
    }
    mods.push_back(std::move(m));
  }
  return mods;
}

// 解析层索引（如 "model.layers.13." 或 "model.language_model.layers.13.")
static int parse_layer_index(const std::string &name) {
  const std::string a = "model.layers.";
  const std::string b = "model.language_model.layers.";
  size_t p = std::string::npos;
  if ((p = name.find(a)) != std::string::npos) {
    p += a.size();
  } else if ((p = name.find(b)) != std::string::npos) {
    p += b.size();
  } else {
    return -1;
  }
  int val = -1;
  size_t idx = p;
  while (idx < name.size() && isdigit(static_cast<unsigned char>(name[idx]))) { idx++; }
  if (idx > p) {
    try { val = std::stoi(name.substr(p, idx - p)); } catch(...) { val = -1; }
  }
  return val;
}

static bool is_expert_weight(const std::string &name) {
  return name.find(".mlp.experts.") != std::string::npos;
}

static bool is_low_precision_name_hint(const std::string &name) {
  return (name.find("fp8") != std::string::npos) ||
         (name.find("_scale") != std::string::npos) ||
         (name.find("_scale_inv") != std::string::npos) ||
         (name.find("_amax") != std::string::npos) ||
         (name.find("_scales") != std::string::npos);
}

static std::vector<int> topk_indices_from_logits(const ow::nn::TensorPtr &logits, int k) {
  std::vector<int> result;
  if (!logits) return result;
  if (logits->shape.size() != 2 || logits->shape[0] != 1) return result;
  int V = logits->shape[1];
  if (k <= 0) return result;
  if (k > V) k = V;
  using Pair = std::pair<float,int>;
  std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
  for (int i = 0; i < V; ++i) {
    float v = logits->get_as_float_flat((size_t)i);
    if ((int)pq.size() < k) pq.emplace(v, i);
    else if (v > pq.top().first) { pq.pop(); pq.emplace(v, i); }
  }
  result.resize((size_t)k);
  for (int i = k - 1; i >= 0; --i) { result[(size_t)i] = pq.top().second; pq.pop(); }
  return result;
}

int main(int argc, char **argv) {
  std::string vocab_path = "assert/vocab.json"; 
  std::string merges_path = "assert/merges.txt";
  std::string weights_dir;

  if (argc >= 3) {
    vocab_path = argv[1];
    merges_path = argv[2];
  }
  if (argc >= 4) {
    weights_dir = argv[3];
  }
  std::cout << "Assets: " << vocab_path << " | " << merges_path << std::endl;

  try {
    ow::nn::Tokenizer tokenizer(vocab_path, merges_path);
    std::string sample = "Hello, world!";
    auto ids = tokenizer.encode(sample);
    std::cout << "Sample: " << sample << " => tokens: ";
    for (size_t i=0;i<ids.size();++i) {
      std::cout << ids[i] << (i+1<ids.size()?",":"");
    }
    std::cout << " => decode: " << tokenizer.decode(ids) << std::endl;
  } catch (const std::exception &ex) {
    std::cerr << "Tokenizer init failed: " << ex.what() << std::endl;
    return 1;
  }

  // Wrap the weights summary within an outer try/catch to avoid else-parsing issues
  try {
    if (!weights_dir.empty()) {
      ow::nn::SafetensorsLoader loader;
      loader.load_dir(weights_dir);

      auto names = loader.names();
      std::cout << "Loaded " << names.size() << " weights from: " << weights_dir << std::endl;

      std::vector<std::string> embeds;
      std::vector<std::string> linears;
      std::vector<std::string> layernorms;
      std::vector<std::string> others;
      std::vector<std::string> experts;

      for (const auto &nm : names) {
        if (is_expert_weight(nm)) experts.push_back(nm);
        else if (str_contains(nm, "embed_tokens.weight") || str_contains(nm, "wte.weight")) embeds.push_back(nm);
        else if (str_contains(nm, ".weight") && (str_contains(nm, "_proj") || str_contains(nm, "lm_head"))) linears.push_back(nm);
        else if (str_contains(nm, "norm.weight")) layernorms.push_back(nm);
        else others.push_back(nm);
      }

      std::cout << "Summary: Embedding=" << embeds.size()
                << ", Linear=" << linears.size()
                << ", LayerNorm=" << layernorms.size()
                << ", Experts=" << experts.size()
                << ", Other=" << others.size() << std::endl;

      auto print_entry = [&](const std::string &key) {
        const auto *e = loader.get_entry(key);
        if (!e) {
          std::cout << "   " << key << " => (missing)" << std::endl;
          return;
        }
        std::cout << "   " << key << " => [" << e->st_dtype << "] shape=(";
        for (size_t j=0;j<e->shape.size();++j) {
          std::cout << e->shape[j] << (j+1<e->shape.size()?",":"");
        }
        std::cout << ")";
        std::cout << std::endl;
      };

      for (int li = 0; li < 6; ++li) {
        std::string prefA = "model.language_model.layers." + std::to_string(li) + ".self_attn.";
        std::string prefB = "model.layers." + std::to_string(li) + ".self_attn.";
        print_entry(prefA + "q_proj.weight");
        print_entry(prefA + "k_proj.weight");
        print_entry(prefA + "v_proj.weight");
        print_entry(prefA + "o_proj.weight");
        print_entry(prefA + "q_norm.weight");
        print_entry(prefA + "k_norm.weight");
        print_entry(prefB + "q_proj.weight");
        print_entry(prefB + "k_proj.weight");
        print_entry(prefB + "v_proj.weight");
        print_entry(prefB + "o_proj.weight");
        print_entry(prefB + "q_norm.weight");
        print_entry(prefB + "k_norm.weight");
      }
      print_entry("model.embed_tokens.weight");
      print_entry("lm_head.weight");

      int show = (int)std::min<size_t>(10, names.size());
      for (int i = 0; i < show; ++i) {
        const auto &nm = names[i];
        const auto *e = loader.get_entry(nm);
        if (!e) continue;
        std::cout << " - " << nm << " [" << e->st_dtype << "] shape=(";
        for (size_t j=0;j<e->shape.size();++j) {
          std::cout << e->shape[j];
          if (j+1 < e->shape.size()) std::cout << ",";
        }
        std::cout << ")";
        std::cout << std::endl;
      }
    } else {
      std::cout << "No weights directory provided; skipping weights summary." << std::endl;
    }
  } catch (const std::exception &ex) {
    std::cerr << "Weights load failed: " << ex.what() << std::endl;
  }

  // MHA smoke test block to validate reshape/matmul order
  {
    using namespace ow::nn;
    auto ctx = std::make_shared<Context>(4*1024*1024);
    int hidden_size = 8;
    int num_heads = 2;
    int num_kv_heads = 1;
    int seq_len = 4;
    int head_dim = hidden_size / num_heads;

    auto mk = [&](std::vector<int> shape) {
      auto T = Tensor::create(ctx, shape, DType::FLOAT32);
      for (size_t i = 0; i < T->nelements(); ++i) {
        int mod = static_cast<int>(i % 7);
        float v = static_cast<float>(mod - 3) * 0.1f;
        T->set_from_float_flat(i, v);
      }
      return T;
    };

    auto q_proj = mk({hidden_size, hidden_size});
    auto k_proj = mk({hidden_size, num_kv_heads * head_dim});
    auto v_proj = mk({hidden_size, num_kv_heads * head_dim});
    auto o_proj = mk({hidden_size, hidden_size});
    auto qnw = mk({head_dim});
    auto knw = mk({head_dim});

    auto mha = std::make_shared<MultiHeadAttention>(q_proj, k_proj, v_proj, o_proj, qnw, knw, num_heads, num_kv_heads, hidden_size);

    auto hidden = mk({seq_len, hidden_size});
    // Sanity: print hidden first row before MHA
    std::cout << "[Sanity] hidden dtype=" << (int)hidden->dtype << " row0: ";
    for (int h=0; h<hidden_size; ++h) {
      std::cout << hidden->get_as_float_flat((size_t)h) << (h+1<hidden_size?",":"");
    }
    std::cout << std::endl;
    auto out = mha->forward(hidden, seq_len);
    std::cout << "[MHA Smoke] out shape=(" << out->shape[0] << "," << out->shape[1] << ")" << std::endl;
    std::cout << "[MHA Smoke] first row: ";
    for (int j=0;j<out->shape[1]; ++j) {
      std::cout << out->get_as_float_flat((size_t)j) << (j+1<out->shape[1] ? "," : "");
    }
    std::cout << std::endl;
  }
  return 0;
}
