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
#include "src/memory_optimizer.hpp"
#include "src/logging.hpp"
#include "include/tokenizer.h"
#include "include/sampler.h"
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
    case ow::nn::DType::FLOAT32: return "F32";
    case ow::nn::DType::FP16: return "F16";
    case ow::nn::DType::BF16: return "BF16";
    case ow::nn::DType::INT32: return "I32";
    case ow::nn::DType::INT8: return "I8";
    case ow::nn::DType::U8: return "U8";
    case ow::nn::DType::BOOL: return "BOOL";
    case ow::nn::DType::I16: return "I16";
    case ow::nn::DType::I64: return "I64";
    case ow::nn::DType::F64: return "F64";
    case ow::nn::DType::FP8_E4M3: return "F8_E4M3";
    case ow::nn::DType::FP8_E5M2: return "F8_E5M2";
    default: return "?";
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

static int parse_layer_index(const std::string &name) {
  size_t pos = name.find(".layers.");
  if (pos == std::string::npos) return -1;
  pos += 8; // skip ".layers."
  std::string num;
  while (pos < name.size() && std::isdigit((unsigned char)name[pos])) num.push_back(name[pos++]);
  if (num.empty()) return -1;
  return std::stoi(num);
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
        if (!std::isfinite(v)) continue;
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
    
    float hidden_max = 0.0f;
    for (int k = 0; k < K; ++k) {
        float val = hidden->get_as_float_flat(k);
        if (std::isfinite(val)) {
            hidden_max = std::max(hidden_max, std::fabs(val));
        }
    }
    
    for (int j = 0; j < V; ++j) {
        double acc = 0.0;
        size_t baseE = (size_t)j * (size_t)K;
        for (int k = 0; k < K; ++k) {
            float a = hidden->get_as_float_flat(k);
            float b = E->get_as_float_flat(baseE + k);
            if (!std::isfinite(a)) a = 0.0f;
            if (!std::isfinite(b)) b = 0.0f;
            double prod = static_cast<double>(a) * static_cast<double>(b);
            if (!std::isfinite(prod)) prod = 0.0;
            acc += prod;
        }
        float result = static_cast<float>(acc);
        if (!std::isfinite(result)) result = 0.0f;
        out->set_from_float_flat(j, result);
    }
    return out;
}

static std::string normalize_weight_name(const std::string &name) {
  if (name.rfind("model.language_model", 0) == 0) return name;
  if (name.rfind("model.", 0) == 0) return name;
  if (name == "embed_tokens.weight") return std::string("model.embed_tokens.weight");
  if (name == "lm_head.weight") return std::string("lm_head.weight");
  return name;
}

// 规范化 token 字符串：将常见分词标记替换为空格
static std::string normalize_token_str(std::string s) {
    const std::string sp_underscore = std::string("\xE2\x96\x81"); // U+2581
    const std::string gpt_space     = std::string("\xC4\xA0");     // U+0120
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (i + sp_underscore.size() <= s.size() && s.compare(i, sp_underscore.size(), sp_underscore) == 0) {
            out.push_back(' ');
            i += sp_underscore.size();
            continue;
        }
        if (i + gpt_space.size() <= s.size() && s.compare(i, gpt_space.size(), gpt_space) == 0) {
            out.push_back(' ');
            i += gpt_space.size();
            continue;
        }
        out.push_back(s[i]);
        ++i;
    }
    return out;
}

// 对 logits 进行 argmax（带数值稳健）
static int argmax_logits(const ow::nn::TensorPtr &logits) {
    int V = logits->shape[1];
    float maxv = -std::numeric_limits<float>::infinity();
    int maxi = 0;
    int valid_count = 0;
    
    for (int i = 0; i < V; ++i) {
        float v = logits->get_as_float_flat(i);
        if (!std::isfinite(v)) continue;
        valid_count++;
        if (v > maxv) { 
            maxv = v; 
            maxi = i; 
        }
    }
    
    if (valid_count == 0) {
        std::cerr << "[WARNING] No valid logits found, using token 0" << std::endl;
        return 0;
    }
    
    return maxi;
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

  // Generation flags
  bool greedy = false;
  bool verbose = false;
  int verbose_every = 10;
  float temperature = 0.7f;
  float top_p = 0.9f;
  uint32_t top_k = 50;
  float repetition_penalty = 1.0f; // 1.0 means disabled

  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--greedy") greedy = true;
    else if (arg == "--verbose") verbose = true;
    else if (arg.rfind("--verbose_every=", 0) == 0) {
      verbose_every = std::max(1, std::stoi(arg.substr(17)));
    } else if (arg.rfind("--temperature=", 0) == 0) {
      temperature = std::stof(arg.substr(14));
    } else if (arg.rfind("--top_p=", 0) == 0) {
      top_p = std::stof(arg.substr(8));
    } else if (arg.rfind("--top_k=", 0) == 0) {
      top_k = (uint32_t)std::stoul(arg.substr(8));
    } else if (arg.rfind("--repetition_penalty=", 0) == 0) {
      repetition_penalty = std::stof(arg.substr(22));
    }
  }

  ow::nn::log(ow::nn::LogLevel::INFO, "SYS", std::string("Assets: ") + vocab_path + " | " + merges_path);

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
    ow::nn::log(ow::nn::LogLevel::ERROR, "SYS", std::string("Tokenizer init failed: ") + ex.what());
    return 1;
  }

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

      {
        std::ostringstream ss;
        ss << "Summary: Embedding=" << embeds.size()
           << ", Linear=" << linears.size()
           << ", LayerNorm=" << layernorms.size()
           << ", Experts=" << experts.size()
           << ", Other=" << others.size();
        ow::nn::log(ow::nn::LogLevel::INFO, "SYS", ss.str());
      }
      
      // Optional Python-like model struct summary
      if (const char* vv = std::getenv("OWNN_VERBOSE_MODEL"); vv && vv[0] != '0') {
        ow::nn::log(ow::nn::LogLevel::INFO, "SYS", "------------ model struct -----------");
        // Accumulate parameter counts by major groups
        std::unordered_map<std::string, size_t> group_params;
        auto names_all = loader.names();
        for (const auto &nm : names_all) {
          const auto *e = loader.get_entry(nm);
          if (!e) continue;
          size_t params = 1;
          for (auto d : e->shape) params *= (size_t)d;
          auto add = [&](const std::string &g){ group_params[g] += params; };
          if (nm.rfind("lm_head.", 0) == 0 || nm == "lm_head.weight") add("lm_head");
          if (nm.rfind("model.", 0) == 0) add("model");
          if (nm.rfind("model.language_model", 0) == 0) add("model.language_model");
          if (nm.find("model.language_model.embed_tokens") != std::string::npos) add("model.language_model.embed_tokens");
          // Per-layer
          size_t pos = nm.find("model.language_model.layers.");
          if (pos != std::string::npos) {
            size_t p2 = pos + std::string("model.language_model.layers.").size();
            std::string idx;
            while (p2 < nm.size() && std::isdigit((unsigned char)nm[p2])) idx.push_back(nm[p2++]);
            if (!idx.empty()) {
              add(std::string("model.language_model.layers.") + idx);
              if (nm.find(".self_attn.", p2) != std::string::npos) add(std::string("layers.") + idx + ".self_attn");
              if (nm.find(".mlp.", p2) != std::string::npos) add(std::string("layers.") + idx + ".mlp");
            }
          }
        }
        auto print_group = [&](const std::string &g){
          auto it = group_params.find(g);
          if (it != group_params.end()) {
            ow::nn::log(ow::nn::LogLevel::INFO, "SYS", g + std::string(" ") + ow::nn::format_params_count(it->second));
          }
        };
        print_group("lm_head");
        print_group("model");
        print_group("model.language_model");
        print_group("model.language_model.embed_tokens");
        // Print a few layer summaries
        for (int li = 0; li < 4; ++li) {
          std::string L = std::string("model.language_model.layers.") + std::to_string(li);
          print_group(L);
          print_group(std::string("layers.") + std::to_string(li) + ".self_attn");
          print_group(std::string("layers.") + std::to_string(li) + ".mlp");
        }
      }

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
      ow::nn::log(ow::nn::LogLevel::WARN, "SYS", "No weights directory provided; skipping weights summary.");
    }
  } catch (const std::exception &ex) {
    ow::nn::log(ow::nn::LogLevel::ERROR, "SYS", std::string("Weights load failed: ") + ex.what());
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
    if (ow::nn::ow_verbose_mha()) {
      std::ostringstream ss;
      ss << "hidden dtype=" << (int)hidden->dtype << " row0: ";
      for (int h=0; h<hidden_size; ++h) {
        ss << hidden->get_as_float_flat((size_t)h) << (h+1<hidden_size?",":"");
      }
      ow::nn::log(ow::nn::LogLevel::DEBUG, "MHA", ss.str());
    }
    auto out = mha->forward(hidden, seq_len);
    if (ow::nn::ow_verbose_mha()) {
      ow::nn::log(ow::nn::LogLevel::DEBUG, "MHA", std::string("out shape=(") + std::to_string(out->shape[0]) + "," + std::to_string(out->shape[1]) + ")");
      std::ostringstream ss2;
      ss2 << "first row: ";
      for (int j=0;j<out->shape[1]; ++j) {
        ss2 << out->get_as_float_flat((size_t)j) << (j+1<out->shape[1] ? "," : "");
      }
      ow::nn::log(ow::nn::LogLevel::DEBUG, "MHA", ss2.str());
    }
  }

  // === Build model and run a short text generation with real weights ===
  try {
    std::string model_dir = weights_dir.empty() ? std::string(DEFAULT_WEIGHTS_DIR) : weights_dir;
    ow::nn::log(ow::nn::LogLevel::INFO, "GEN", std::string("Using model dir: ") + model_dir);

    ow::nn::LazySafetensorsLoader gen_loader;
    gen_loader.load_dir(model_dir);
    
    size_t optimal_size = ow::nn::MemoryOptimizer::get_optimal_initial_context_size();
    auto gen_ctx = std::make_shared<ow::nn::Context>(optimal_size);
    
    {
      std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(2);
      ss << "Using optimized context size: " << (optimal_size / (1024.0 * 1024.0 * 1024.0)) << " GB";
      ow::nn::log(ow::nn::LogLevel::INFO, "GEN", ss.str());
    }
    
    std::vector<std::string> weight_patterns = {
        "model.language_model",
        "language_model",
        "model.layers",
        "layers.",
        "model.embed_tokens",
        "embed_tokens",
        "wte",
        "model.norm",
        "norm.weight",
        "lm_head"
    };
    
    auto all_weights = ow::nn::MemoryOptimizer::load_weights_optimized(
        gen_loader, gen_ctx, weight_patterns, true
    );

    ow::nn::WeightLoader wl;
    for (const auto &kv : all_weights) {
        wl.add_weight(normalize_weight_name(kv.first), kv.second);
    }
    auto model = wl.build_model();

    std::string vocab_path_m = model_dir + "\\vocab.json";
    std::string merges_path_m = model_dir + "\\merges.txt";
    std::string vocab_path_final = vocab_path;
    std::string merges_path_final = merges_path;
    if (std::filesystem::exists(vocab_path_m) && std::filesystem::exists(merges_path_m)) {
      vocab_path_final = vocab_path_m;
      merges_path_final = merges_path_m;
    }
    ow::nn::Tokenizer tok(vocab_path_final, merges_path_final);

    ow::nn::log(ow::nn::LogLevel::INFO, "Chat", "Type your question. Use /exit to quit.");
    std::vector<std::pair<std::string, std::string>> history;
    while (true) {
      std::string question;
      std::cout << "Q> ";
      if (!std::getline(std::cin, question)) break;
      if (question == "/exit") break;
      if (question.empty()) continue;

      std::string prompt_text;
      for (auto &turn : history) {
        prompt_text += "User: " + turn.first + "\n";
        prompt_text += "Assistant: " + turn.second + "\n";
      }
      prompt_text += "User: " + question + "\nAssistant: ";

      auto ids = tok.encode(prompt_text);
      int max_new_tokens = 64;
      std::string answer;
      int consecutive_empty_tokens = 0;
      int nextTokenId = -1;
      bool placeholder = false;

      sampler sp;
      sp.temperature = temperature;
      sp.top_p = top_p;
      sp.top_k = top_k;
      sp.repetition_penalty = repetition_penalty;
      sp.apply_softmax = true;
      sp.do_sample = !greedy;

      for (int step = 0; step < max_new_tokens; ++step) {
        auto hidden_states = model->forward(ids);
        
        bool has_nan = false;
        size_t n = hidden_states->nelements();
        for (size_t i = 0; i < std::min(n, size_t(100)); ++i) {
          if (std::isnan(hidden_states->get_as_float_flat(i))) {
            has_nan = true;
            break;
          }
        }
        if (has_nan) {
          ow::nn::log(ow::nn::LogLevel::ERROR, "GEN", std::string("NaN detected in hidden states at step ") + std::to_string(step) + ", stopping generation");
          break;
        }
        
        int seq_len2 = (int)ids.size();
        int hidden_size2 = hidden_states->shape[1];
        auto last_hidden = hidden_states->slice_view({seq_len2 - 1, 0}, {1, hidden_size2});

        ow::nn::TensorPtr vocab_weight = wl.get_weight("lm_head.weight");
        if (!vocab_weight) vocab_weight = wl.get_weight("model.lm_head.weight");
        if (!vocab_weight) vocab_weight = wl.get_weight("model.embed_tokens.weight");
        if (!vocab_weight) throw std::runtime_error("No vocab projection weight found");

        auto logits = ow::nn::Tensor::matvec_blocked_mt(last_hidden, vocab_weight);
        
        has_nan = false;
        n = logits->nelements();
        for (size_t i = 0; i < n; ++i) {
          if (std::isnan(logits->get_as_float_flat(i))) {
            has_nan = true;
            break;
          }
        }
        if (has_nan) {
          ow::nn::log(ow::nn::LogLevel::ERROR, "GEN", std::string("NaN detected in logits at step ") + std::to_string(step) + ", stopping generation");
          break;
        }
        
        if (verbose || ow::nn::ow_verbose_gen()) {
          if (step % verbose_every == 0) {
            auto top5 = topk_indices_from_logits(logits, 5);
            std::ostringstream oss;
            oss << "step=" << step << " top-5: ";
            for (const auto &p : top5) { oss << "(" << p.first << ":" << p.second << ") "; }
            ow::nn::log(ow::nn::LogLevel::DEBUG, "GEN", oss.str());
          }
        }

        if (greedy) {
          nextTokenId = argmax_logits(logits);
        } else {
          int V = logits->shape[1];
          sp.vocab_size = (uint32_t)V;
          std::vector<float> logit_vec(V);
          for (int i = 0; i < V; ++i) logit_vec[i] = logits->get_as_float_flat(i);
          std::vector<uint32_t> out_ids;
          sp.sample(logit_vec.data(), out_ids);
          if (out_ids.empty()) {
            nextTokenId = argmax_logits(logits);
          } else {
            nextTokenId = (int)out_ids.back();
          }
        }
        ids.push_back(nextTokenId);

        std::string piece_raw = tok.decode_id(nextTokenId);
        std::string piece = normalize_token_str(piece_raw);

        if (verbose || ow::nn::ow_verbose_gen()) {
          if (step % verbose_every == 0) {
            ow::nn::log(ow::nn::LogLevel::DEBUG, "GEN", std::string("tok id=") + std::to_string(nextTokenId) + " raw=\"" + piece_raw + "\"");
          }
        }

        if (piece == "</s>" || piece == " ") {
          ow::nn::log(ow::nn::LogLevel::INFO, "GEN", std::string("EOS reached."));
          std::cout << "\n";
          break;
        }

        std::cout << piece << std::flush;
        answer += piece;

        placeholder = piece.size() >= 4 && piece.find("<id=") != std::string::npos;
        if (!placeholder && (piece.empty() || piece == " " || piece == "\t")) {
          consecutive_empty_tokens++;
          if (consecutive_empty_tokens > 20) {
            ow::nn::log(ow::nn::LogLevel::WARN, "GEN", std::string("Too many consecutive empty tokens, stopping generation"));
            break;
          }
        } else {
          consecutive_empty_tokens = 0;
        }

        if (!piece.empty() && piece.back() == '\n') { break; }
      }
      std::cout << std::endl;
      history.emplace_back(question, answer);
    }
  } catch (const std::exception &e) {
    ow::nn::log(ow::nn::LogLevel::ERROR, "GEN", std::string("Error: ") + e.what());
  }

  return 0;
}
