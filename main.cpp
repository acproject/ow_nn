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
#include <algorithm>
 
 #include "include/ow_nn.h"
#include "include/transformer.h"
#include "src/safetensors_loader.hpp"
#include "src/lazy_safetensors_loader.hpp"
#include "src/context.hpp"
#include "src/tensor.hpp"
#include "include/tokenizer.h"
#include <unordered_map>

// Default asset and weight paths for local testing
static const char* DEFAULT_WEIGHTS_DIR = "d:\\workspace\\cpp_projects\\ow_nn\\model";
static const char* DEFAULT_VOCAB_PATH  = "d:\\workspace\\cpp_projects\\ow_nn\\model\\vocab.json";
static const char* DEFAULT_MERGES_PATH = "d:\\workspace\\cpp_projects\\ow_nn\\model\\merges.txt";

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

// Helper: compute top-k indices from logits for debugging
std::vector<std::pair<int, float>> topk_indices_from_logits(const ow::nn::TensorPtr &logits, int k) {
    std::vector<std::pair<int, float>> topk;
    int vocab = logits->shape[1];
    for (int i = 0; i < vocab; ++i) {
        float v = logits->get_as_float_flat(i);
        if ((int)topk.size() < k) {
            topk.emplace_back(i, v);
            if ((int)topk.size() == k) {
                std::sort(topk.begin(), topk.end(), [](auto &a, auto &b) { return a.second > b.second; });
            }
        } else if (v > topk.back().second) {
            topk.back() = {i, v};
            std::sort(topk.begin(), topk.end(), [](auto &a, auto &b) { return a.second > b.second; });
        }
    }
    return topk;
}

// Row-oriented matvec: logits = hidden[1,K] dot each row of E[V,K] -> [1,V]
static ow::nn::TensorPtr matvec_rows_dot(const ow::nn::TensorPtr &hidden, const ow::nn::TensorPtr &E) {
    using namespace ow::nn;
    if (!hidden || !E) throw std::runtime_error("matvec_rows_dot: null tensor(s)");
    if (hidden->shape.size() != 2 || E->shape.size() != 2) throw std::runtime_error("matvec_rows_dot: expect 2D tensors");
    int K = hidden->shape[1];
    int V = E->shape[0];
    int Ek = E->shape[1];
    if (K != Ek) throw std::runtime_error("matvec_rows_dot: inner dim mismatch");
    auto ctx = hidden->ctx.lock();
    if (!ctx) throw std::runtime_error("matvec_rows_dot: missing context");
    auto out = ow::nn::Tensor::create(ctx, {1, V}, ow::nn::DType::FLOAT32);
    // Compute logits row-by-row against embedding/lm_head rows
    for (int j = 0; j < V; ++j) {
        float acc = 0.0f;
        size_t baseE = (size_t)j * (size_t)K;
        for (int k = 0; k < K; ++k) {
            float a = hidden->get_as_float_flat(k);
            float b = E->get_as_float_flat(baseE + k);
            acc += a * b;
        }
        out->set_from_float_flat(j, acc);
    }
    return out;
}

// Normalize weight names: strip 'model.language_model.' to 'model.' if present
static std::string normalize_weight_name(const std::string &name) {
    const std::string prefix = "model.language_model.";
    if (name.rfind(prefix, 0) == 0) {
        std::string normalized = std::string("model.") + name.substr(prefix.size());
        std::cout << "[WeightMap] " << name << " -> " << normalized << std::endl;
        return normalized;
    }
    return name;
}

int main(int argc, char **argv) {
  std::string vocab_path = DEFAULT_VOCAB_PATH; 
  std::string merges_path = DEFAULT_MERGES_PATH;
  std::string weights_dir = DEFAULT_WEIGHTS_DIR;

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
    auto ctx = std::make_shared<Context>(16*1024*1024); // 16MB for test
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

  // === Build model and run a short text generation with real weights ===
  try {
    std::string model_dir = weights_dir.empty() ? std::string(DEFAULT_WEIGHTS_DIR) : weights_dir;
    std::cout << "[GEN] Using model dir: " << model_dir << "\n";

    ow::nn::LazySafetensorsLoader gen_loader;
    gen_loader.load_dir(model_dir);
    // Use 64MB initial arena size, will grow dynamically as needed
    auto gen_ctx = std::make_shared<ow::nn::Context>(64ull * 1024ull * 1024ull);

    // Filter and load text-only weights to reduce memory usage
    // Use lazy loading with copy=true for large files to avoid keeping them mapped
    std::unordered_map<std::string, ow::nn::TensorPtr> all_weights;
    std::cout << "[GEN] Loading weights with lazy strategy...\n";
    
    for (const auto &name : gen_loader.names()) {
        // Load all model weights including language_model, embed_tokens, lm_head, and norm
        if (name.rfind("model.language_model.", 0) == 0 ||
            name.rfind("model.embed_tokens.", 0) == 0 ||
            name.rfind("model.norm.", 0) == 0 ||
            name == "model.embed_tokens.weight" ||
            name == "lm_head.weight" ||
            name == "model.lm_head.weight" ||
            name == "model.norm.weight") {
            try {
                // Use copy=true to avoid keeping large files mapped
                auto t = gen_loader.make_tensor(name, gen_ctx, /*copy=*/true);
                if (t) {
                    all_weights.emplace(name, t);
                    std::cout << "[Weight][loaded] " << name << " shape: [";
                    for (size_t i = 0; i < t->shape.size(); i++) {
                        std::cout << t->shape[i];
                        if (i < t->shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
            } catch (const std::exception &ex) {
                std::cerr << "[Weight][skip] " << name << " due to: " << ex.what() << "\n";
            }
        }
    }

    ow::nn::WeightLoader wl;
    for (const auto &kv : all_weights) {
        wl.add_weight(normalize_weight_name(kv.first), kv.second);
    }
    auto model = wl.build_model();

    // Prefer tokenizer in model dir, fallback to assets
    std::string vocab_path_m = model_dir + "\\vocab.json";
    std::string merges_path_m = model_dir + "\\merges.txt";
    std::string vocab_path_final = vocab_path;
    std::string merges_path_final = merges_path;
    if (std::filesystem::exists(vocab_path_m) && std::filesystem::exists(merges_path_m)) {
      vocab_path_final = vocab_path_m;
      merges_path_final = merges_path_m;
    }
    ow::nn::Tokenizer tok(vocab_path_final, merges_path_final);

    std::cout << "[Chat] Type your question. Use /exit to quit.\n";
    std::vector<std::pair<std::string, std::string>> history;
    while (true) {
      std::string question;
      std::cout << "Q> ";
      if (!std::getline(std::cin, question)) break;
      if (question == "/exit") break;
      if (question.empty()) continue;

      // Build simple Q&A-style prompt with minimal history
      std::string prompt_text;
      for (auto &turn : history) {
        prompt_text += "User: " + turn.first + "\n";
        prompt_text += "Assistant: " + turn.second + "\n";
      }
      prompt_text += "User: " + question + "\nAssistant: ";

      auto ids = tok.encode(prompt_text);
      int max_new_tokens = 64;
      std::string answer;
      for (int step = 0; step < max_new_tokens; ++step) {
        auto hidden_states = model->forward(ids);
        int seq_len2 = (int)ids.size();
        int hidden_size2 = hidden_states->shape[1];
        auto last_hidden = hidden_states->slice_view({seq_len2 - 1, 0}, {1, hidden_size2});

        ow::nn::TensorPtr vocab_weight = wl.get_weight("lm_head.weight");
        if (!vocab_weight) vocab_weight = wl.get_weight("model.lm_head.weight");
        if (!vocab_weight) vocab_weight = wl.get_weight("model.embed_tokens.weight");
        if (!vocab_weight) throw std::runtime_error("No vocab projection weight found");

        auto logits = matvec_rows_dot(last_hidden, vocab_weight);
        // Optional: print top-5 for debugging
        auto top5 = topk_indices_from_logits(logits, 5);
        std::cout << "[DBG top-5] ";
        for (auto &p : top5) std::cout << "(" << p.first << ":" << p.second << ") ";
        std::cout << "\n";

        auto next = std::max_element(top5.begin(), top5.end(), [](auto &a, auto &b){return a.second < b.second;});
        int next_id = next->first;
        ids.push_back(next_id);
        std::string piece = tok.decode({next_id});
        std::cout << piece << std::flush;
        answer += piece;

        if (!piece.empty() && piece.back() == '\n') break;
      }
      std::cout << std::endl;
      history.emplace_back(question, answer);
    }
  } catch (const std::exception &e) {
    std::cerr << "[GEN][Error] " << e.what() << "\n";
  }

  return 0;
}
