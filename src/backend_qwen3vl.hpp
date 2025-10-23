#pragma once
#include "context.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ow::nn {

struct RopeParameters {
  float rope_theta;
  std::optional<std::string> rope_type;
  std::optional<float> factor;
  std::optional<int> original_max_position_embeddings;
  std::optional<float> attention_factor;
  std::optional<float> beta_fast;
  std::optional<float> beta_slow;
  std::optional<std::vector<float>> short_factor;
  std::optional<std::vector<float>> long_factor;
  std::optional<float> low_freq_factor;
  std::optional<float> high_freq_factor;
  std::optional<std::unordered_map<std::string, std::vector<int>>>
      mrope_section;
};

static inline bool has_mrope_section(
    const std::optional<std::variant<
        RopeParameters, std::unordered_map<std::string, RopeParameters>>>
        &rope_opt) {
  if (!rope_opt.has_value()) {
    return false; // rope_parameters 是 nullopt
  }

  return std::visit(
      [](const auto &val) -> bool {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, RopeParameters>) {
          // 是单个 RopeParameters，检查其 mrope_section
          return val.mrope_section.has_value();
        } else if constexpr (std::is_same_v<
                                 T, std::unordered_map<std::string,
                                                       RopeParameters>>) {
          // 是 map，通常 mrope_section 只在具体的 RopeParameters 实例中
          // 这里需要定义“是否有”：比如任意一个 value 有？还是都不算？
          // 通常：map 本身不包含 mrope_section，而是每个 value 有
          // 所以你可以选择：
          //   - 检查所有项
          //   - 或认为 map 类型下“没有全局 mrope_section”
          // 根据你的语义决定！

          // 示例：只要有一个 RopeParameters 有 mrope_section 就返回 true
          for (const auto &[key, rp] : val) {
            if (rp.mrope_section.has_value()) {
              return true;
            }
          }
          return false;
        } else {
          return false; // 不可能走到这里
        }
      },
      rope_opt.value());
}

struct Qwen3VLTextConfig {
  int vocab_size = 151936;
  int hidden_size = 4096;
  int intermediate_size = 22016;
  int num_hidden_layers = 32;
  int num_attention_heads = 32;
  int num_key_value_heads = 32;
  int head_dim = 128;
  std::string hidden_act = "silu";
  int max_position_embeddings = 128000;
  double initializer_range = 0.02;
  double rms_norm_eps = 1e-6;
  bool use_cache = true;
  bool tie_word_embedding = false;

  std::optional<std::variant<RopeParameters,
                             std::unordered_map<std::string, RopeParameters>>>
      rope_parameters = std::nullopt;
  bool attention_bias = false;
  std::optional<std::vector<std::string>> layer_types = std::nullopt;
  double attention_dropout = 0.0;

  std::string model_type = "qwen3_vl";
  std::string base_config_key = "vision_config";

  Qwen3VLTextConfig() = default;
};

struct Qwen3VLVisionConfig {
  static inline const std::string model_type = "qwen3_vl";
  static inline const std::string base_config_key = "vision_config";
  int depth = 27;
  int hidden_size = 1152;
  std::string hidden_act = "gelu_pytorch_tahn";
  int intermediate_size = 4304;
  int num_heads = 16;
  int in_channels = 3;
  int patch_size = 16;
  int spatial_merge_size = 16;
  int temporal_patch_size = 2;
  int out_hidden_size = 3584;
  int num_position_emeddings = 2304;
  std::vector<int> deepstack_visual_indexes{8, 16, 24};
  float initializer_range = 0.02;
};

struct Qwen3VLConfig {
  static inline const std::string model_type = "qwen3_vl";
  static inline const std::unordered_map<std::string, std::string> sub_config =
      {{"vision_config", "Qwen3VLVisionConfig"},
       {"text_config", "Qwen3VLTextConfig"}};
  static inline const std::vector<std::string> keys_to_ignore_at_inference = {
      "past_key_values"};

  Qwen3VLTextConfig text_config{};
  Qwen3VLVisionConfig vision_config{};
  int image_token_id = 151655;
  int video_token_id = 151656;
  int vision_start_token_id = 151652;
  int vision_end_token_id = 151653;
  bool tie_word_embeddings = false;

  Qwen3VLConfig() = default;
};

// Qwen2RMSNorm 等价实现（与 T5LayerNorm 等价）
struct Qwen2RMSNorm {
  TensorPtr weight; // shape: [hidden_size]
  float variance_epsilon = 1e-6f;

  Qwen2RMSNorm(int hidden_size, float eps, const std::shared_ptr<Context> &ctx)
      : variance_epsilon(eps) {
    if (!ctx)
      throw std::runtime_error("Qwen2RMSNorm: ctx expired");
    weight = Tensor::ones(ctx, {hidden_size}, DType::FLOAT32);
    size_t n = weight->nelements();
    // for (size_t i = 0; i < n; ++i)  // 因为采用了ones，所以这里不需要了
    //   weight->set_from_float_flat(i, 1.0f);
  }

  // 前向传播：对输入 hidden_states 执行 RMSNorm 归一化
  // 公式：y = x / RMS(x) * weight，其中 RMS(x) = sqrt(mean(x^2) + eps)
  TensorPtr forward(const TensorPtr &hidden_states) const {
    // 参数校验
    if (!hidden_states || !weight)
      throw std::runtime_error("Qwen2RMSNorm: null input or weight");
    auto ctx = hidden_states->ctx.lock();
    if (!ctx)
      throw std::runtime_error("Qwen2RMSNorm: ctx expired");
    auto shape = hidden_states->shape;
    if (shape.empty())
      throw std::runtime_error("Qwen2RMSNorm: input rank must be >= 1");

    // 获取最后一维作为 hidden_size，其余维度合并为 batch_seq
    int hidden_size = shape.back();
    size_t batch_seq = hidden_states->nelements() / hidden_size;
    auto in_dtype = hidden_states->dtype;

    // 创建输出张量，统一使用 float32 进行计算
    auto Y = Tensor::create(ctx, shape, DType::FLOAT32);

    // 遍历每个 token（或 patch）向量
    for (size_t i = 0; i < batch_seq; ++i) {
      // 1. 计算该向量元素的平方均值
      float mean_sq = 0.0f;
      for (int j = 0; j < hidden_size; ++j) {
        float v = hidden_states->get_as_float_flat(i * hidden_size + j);
        mean_sq += v * v;
      }
      mean_sq /= (float)hidden_size;

      // 2. 计算逆 RMS 值，加入 epsilon 防止除零
      float inv_rms = 1.0f / std::sqrt(mean_sq + variance_epsilon);

      // 3. 归一化后乘以可学习权重，写入输出
      for (int j = 0; j < hidden_size; ++j) {
        float v =
            hidden_states->get_as_float_flat(i * hidden_size + j) * inv_rms;
        float w = weight->get_as_float_flat(j);
        Y->set_from_float_flat(i * hidden_size + j, v * w);
      }
    }

    // 如果原始 dtype 不是 float32，则转换回去
    if (in_dtype != DType::FLOAT32) {
      return Y->astype(in_dtype);
    }
    return Y;
  }

  std::string extra_repr() const {
    return std::string("(") + std::to_string(weight ? weight->shape[0] : 0) +
           ")" + ", eps=" + std::to_string(variance_epsilon);
  }
};

struct LlamaConfig {
  static inline const std::string model_type = "llama";
  static inline const std::vector<std::string> key_to_ignore_at_inference = {
      "past_key_values"};
  static inline const std::unordered_map<std::string, std::string>
      base_model_tp_plan = {
          {"layers.*.self_attn.q_proj", "colwise"},
          {"layers.*.self_attn.k_proj", "colwise"},
          {"layers.*.self_attn.v_proj", "colwise"},
          {"layers.*.self_attn.o_proj", "rowwise"},
          {"layers.*.mlp.gate_proj", "colwise"},
          {"layers.*.mlp.up_proj", "colwise"},
          {"layers.*.mlp.down_proj", "rowwise"},
  };
  static inline const std::unordered_map<
      std::string,
      std::tuple<std::vector<std::string>, std::vector<std::string>>>
      model_io_map = {
          {"embed_tokens", {{"input_ids"}, {"inputs_embeds"}}},
          {"layers", {{"hidden_states", "attention_mask"}, {"hidden_states"}}},
          {"norm", {{"hidden_states"}, {"hidden_states"}}},
  };

  LlamaConfig() {
    vocab_size = 32000;
    hidden_size = 4096;
    intermediate_size = 11008;
    num_hidden_layers = 32;
    num_attention_heads = 32;
    num_key_value_heads = std::nullopt;
    hidden_act = "silu";
    max_position_embeddings = 2048;
    initializer_range = 0.02;
    rms_norm_eps = 1e-6;
    use_cache = true;
    pad_token_id = std::nullopt;
    bos_token_id = 1;
    eos_token_id = 2;
    pretraining_tp = 1;
    tie_word_embeddings = false;
    rope_parameters = std::nullopt;
    attention_bias = false;
    attention_dropout = 0.0;
    mlp_bias = false;
    head_dim = std::nullopt;
  }

  int vocab_size = 32000;
  int hidden_size = 4096;
  int intermediate_size = 11008;
  int num_hidden_layers = 32;
  int num_attention_heads = 32;
  std::optional<int> num_key_value_heads;
  std::string hidden_act = "silu";
  int max_position_embeddings = 2048;
  double initializer_range = 0.02;
  double rms_norm_eps = 1e-6;
  bool use_cache = true;
  std::optional<int> pad_token_id;
  int bos_token_id = 1;
  int eos_token_id = 2;
  int pretraining_tp = 1;
  bool tie_word_embeddings = false;
  std::optional<std::variant<RopeParameters,
                             std::unordered_map<std::string, RopeParameters>>>
      rope_parameters;
  bool attention_bias = false;
  double attention_dropout = 0.0;
  bool mlp_bias = false;
  std::optional<int> head_dim;
};

struct LlamaRotaryEmbedding {
  int max_seq_len_cached = 0;
  int original_max_seq_len = 0;
  std::string rope_type = "default";
  float attention_scaling = 1.0f;
  TensorPtr inv_freq;
  TensorPtr original_inv_freq;

  static inline std::pair<TensorPtr, float>
  compute_default_rope_parameters(const LlamaConfig &config,
                                  const std::shared_ptr<Context> &ctx) {
    if (!ctx)
      throw std::runtime_error("LlamaRotaryEmbedding: ctx expired");

    float base = 10000.0f;
    // Try to get rope theta from config.rope_parameters
    if (config.rope_parameters.has_value()) {
      const auto &var = config.rope_parameters.value();
      if (std::holds_alternative<RopeParameters>(var)) {
        const auto &rp = std::get<RopeParameters>(var);
        base = rp.rope_theta;
      } else {
        const auto &mp =
            std::get<std::unordered_map<std::string, RopeParameters>>(var);
        if (!mp.empty()) {
          const auto &rp = mp.begin()->second; // pick the first as default
          base = rp.rope_theta;
        }
      }
    }

    int dim = config.head_dim.has_value()
                  ? config.head_dim.value()
                  : (config.hidden_size / config.num_attention_heads);
    if (dim <= 0)
      throw std::runtime_error("LlamaRotaryEmbedding: invalid head dim");
    if ((dim % 2) != 0)
      throw std::runtime_error(
          "LlamaRotaryEmbedding: head dim must be even for RoPE");

    auto inv = Tensor::create(ctx, {dim / 2}, DType::FLOAT32);
    for (int i = 0; i < dim / 2; ++i) {
      float exponent = (2.0f * (float)i) / (float)dim;
      float val = 1.0f / std::pow(base, exponent);
      inv->set_from_float_flat((size_t)i, val);
    }
    float attention_factor = 1.0f; // Unused for default RoPE
    return {inv, attention_factor};
  }

  LlamaRotaryEmbedding(const std::shared_ptr<Context> &ctx,
                       const LlamaConfig &config) {
    if (!ctx)
      throw std::runtime_error("LlamaRotaryEmbedding: ctx expired");

    max_seq_len_cached = config.max_position_embeddings;
    original_max_seq_len = config.max_position_embeddings;

    // rope_type selection (fallback to default if not provided)
    if (config.rope_parameters.has_value()) {
      const auto &var = config.rope_parameters.value();
      if (std::holds_alternative<RopeParameters>(var)) {
        const auto &rp = std::get<RopeParameters>(var);
        if (rp.rope_type.has_value())
          rope_type = rp.rope_type.value();
      } else {
        const auto &mp =
            std::get<std::unordered_map<std::string, RopeParameters>>(var);
        if (!mp.empty()) {
          const auto &rp = mp.begin()->second;
          if (rp.rope_type.has_value())
            rope_type = rp.rope_type.value();
        }
      }
    }

    // Currently only default is implemented; others fallback to default
    auto res = compute_default_rope_parameters(config, ctx);
    inv_freq = res.first;
    attention_scaling = res.second;
    original_inv_freq = inv_freq; // mirror python semantics
  }
};

struct Qwen3VLTextRotaryEmbedding : public LlamaRotaryEmbedding {
  std::vector<int> mrope_section;

  static inline LlamaConfig to_llama_config(const Qwen3VLTextConfig &cfg) {
    LlamaConfig lc;
    lc.vocab_size = cfg.vocab_size;
    lc.hidden_size = cfg.hidden_size;
    lc.intermediate_size = cfg.intermediate_size;
    lc.num_hidden_layers = cfg.num_hidden_layers;
    lc.num_attention_heads = cfg.num_attention_heads;
    lc.num_key_value_heads = cfg.num_key_value_heads;
    lc.hidden_act = cfg.hidden_act;
    lc.max_position_embeddings = cfg.max_position_embeddings;
    lc.initializer_range = cfg.initializer_range;
    lc.rms_norm_eps = cfg.rms_norm_eps;
    lc.use_cache = cfg.use_cache;
    lc.tie_word_embeddings = cfg.tie_word_embedding;
    lc.rope_parameters = cfg.rope_parameters;
    lc.attention_bias = cfg.attention_bias;
    lc.attention_dropout = cfg.attention_dropout;
    lc.head_dim = cfg.head_dim;
    return lc;
  }

  Qwen3VLTextRotaryEmbedding(const std::shared_ptr<Context> &ctx,
                             const Qwen3VLTextConfig &config)
      : LlamaRotaryEmbedding(ctx, to_llama_config(config)) {

    if (has_mrope_section(config.rope_parameters)) {
      if (config.rope_parameters.has_value()) {
        const auto &var = config.rope_parameters.value();
        if (std::holds_alternative<RopeParameters>(var)) {
          const auto &rp = std::get<RopeParameters>(var);
          if (rp.mrope_section.has_value()) {
            const auto &mrope_map = rp.mrope_section.value();
            if (mrope_map.count("mrope_section")) {
              mrope_section = mrope_map.at("mrope_section");
            } else {
              mrope_section = {24, 20, 20};
            }
          } else {
            mrope_section = {24, 20, 20};
          }
        } else {
          const auto &mp =
              std::get<std::unordered_map<std::string, RopeParameters>>(var);
          if (mp.count("mrope_section")) {
            const auto &rp = mp.at("mrope_section");
            if (rp.mrope_section.has_value()) {
              const auto &mrope_map = rp.mrope_section.value();
              if (mrope_map.count("mrope_section")) {
                mrope_section = mrope_map.at("mrope_section");
              } else {
                mrope_section = {24, 20, 20};
              }
            } else {
              mrope_section = {24, 20, 20};
            }
          } else {
            mrope_section = {24, 20, 20};
          }
        }
      } else {
        mrope_section = {24, 20, 20};
      }
    } else {
      mrope_section = {24, 20, 20};
    }
  }

  TensorPtr
  apply_interleaved_mrope(const TensorPtr &freqs,
                          const std::vector<int> &mrope_section) const {
    auto ctx = freqs->ctx.lock();
    if (!ctx)
      throw std::runtime_error("apply_interleaved_mrope: ctx expired");

    if (freqs->shape.size() != 4 || freqs->shape[0] != 3)
      throw std::runtime_error(
          "apply_interleaved_mrope: freqs must be [3,b,seq,dim/2]");

    if (mrope_section.size() != 3)
      throw std::runtime_error("mrope_section must have 3 elements [t,h,w]");

    // 将时间维度（T）的频率张量作为基础模板，后续会在其对应切片上覆盖空间维度（H、W)的频率值
    auto freqs_t = (*freqs)[0]; // 仅保留时间维度的频率进行

    // 遍历空间维度 dim=1(H)、dim=2(W)，并使用 offset=1、2 交错覆盖
    for (int dim = 1, offset = 1; dim <= 2; ++dim, ++offset) {
      /**
       Python 的 slice(start, stop, step)
       里，第二个参数是“终止索引（不含）”，例如 stop = mrope_section[dim] * 3
       ，用 step=3 时会选出 mrope_section[dim] 个位置。 我们的 C++ sl(start,
       length, step)里，第二个参数是“选取的元素个数”， 不是“终止索引”。因此 C++
       里应该使用 length = mrope_section[dim] ，不能乘 3。
       */
      int length = mrope_section[dim]; // 每个维度的组数（按步长 3 取）
      // 源视图：freqs[dim, ..., offset:offset+length*3:3]，即步进3的切片
      auto src = freqs->at({idx(dim), ellipsis(), sl(offset, length, 3)});
      // 目标赋值：将 src 交错写入 freqs_t 的最后一维
      freqs_t->slice_assign(-1, offset, length, 3, src);
    }

    return freqs_t;
  }

  std::pair<TensorPtr, TensorPtr> forward(const TensorPtr &x,
                                          const TensorPtr &position_ids) const {
    if (!x || !position_ids)
      throw std::runtime_error(
          "Qwen3VLTextRotaryEmbedding::forward: null input");

    auto ctx = get_ctx_or_throw(x);

    TensorPtr pos3;
    if (position_ids->shape.size() == 2) {
      int B = position_ids->shape[0];
      int S = position_ids->shape[1];
      pos3 = Tensor::create(ctx, {3, B, S}, DType::FLOAT32);
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          float v = position_ids->get_as_float_flat((size_t)b * (size_t)S +
                                                    (size_t)s);
          pos3->set_from_float_flat((size_t)0 * (size_t)B * (size_t)S +
                                        (size_t)b * (size_t)S + (size_t)s,
                                    v);
          pos3->set_from_float_flat((size_t)1 * (size_t)B * (size_t)S +
                                        (size_t)b * (size_t)S + (size_t)s,
                                    v);
          pos3->set_from_float_flat((size_t)2 * (size_t)B * (size_t)S +
                                        (size_t)b * (size_t)S + (size_t)s,
                                    v);
        }
      }
    } else if (position_ids->shape.size() == 3 && position_ids->shape[0] == 3) {
      pos3 = position_ids;
    } else {
      throw std::runtime_error("Qwen3VLTextRotaryEmbedding::forward: "
                               "position_ids must be [B,S] or [3,B,S]");
    }

    int B = pos3->shape[1];
    int S = pos3->shape[2];
    int half_dim = (int)inv_freq->shape[0];

    auto freqs = Tensor::create(ctx, {3, B, S, half_dim}, DType::FLOAT32);
    for (int dim_idx = 0; dim_idx < 3; ++dim_idx) {
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          float p =
              pos3->get_as_float_flat((size_t)dim_idx * (size_t)B * (size_t)S +
                                      (size_t)b * (size_t)S + (size_t)s);
          for (int i = 0; i < half_dim; ++i) {
            float iv = inv_freq->get_as_float_flat((size_t)i);
            float v = iv * p;
            size_t off =
                (size_t)dim_idx * (size_t)B * (size_t)S * (size_t)half_dim +
                (size_t)b * (size_t)S * (size_t)half_dim +
                (size_t)s * (size_t)half_dim + (size_t)i;
            freqs->set_from_float_flat(off, v);
          }
        }
      }
    }

    auto freqs_t = apply_interleaved_mrope(freqs, mrope_section);

    auto emb = concat({freqs_t, freqs_t}, -1);

    auto cos = emb->elementwise_unary(emb, [this](float vv) {
      return std::cos(vv) * this->attention_scaling;
    });
    auto sin = emb->elementwise_unary(emb, [this](float vv) {
      return std::sin(vv) * this->attention_scaling;
    });

    return {cos, sin};
  }
};

} // namespace ow::nn