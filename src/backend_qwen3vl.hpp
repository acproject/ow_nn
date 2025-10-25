#pragma once
#include "context.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
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
  std::variant<TensorPtr, std::string> hidden_act = "silu";
  int max_position_embeddings = 128000;
  double initializer_range = 0.02;
  double rms_norm_eps = 1e-6;
  bool use_cache = true;
  bool tie_word_embedding = false;
  // Q/K RMSNorm 默认启用（符合 Qwen3 官方实现）
  bool use_qk_rmsnorm = true;
  // 控制是否应用因果掩码，默认启用（符合大多数解码器自注意力）
  bool is_causal = true;

  std::optional<std::variant<RopeParameters,
                             std::unordered_map<std::string, RopeParameters>>>
      rope_parameters = std::nullopt;
  bool attention_bias = false;
  std::optional<std::vector<std::string>> layer_types = std::nullopt;
  double attention_dropout = 0.0;
  std::optional<int> sliding_window = std::nullopt;

  std::string model_type = "qwen3_vl";
  std::string base_config_key = "vision_config";

  Qwen3VLTextConfig() = default;
};

struct Qwen3VLVisionConfig {
  static inline const std::string model_type = "qwen3_vl";
  static inline const std::string base_config_key = "vision_config";
  int depth = 27;
  int hidden_size = 1152;
  std::variant<TensorPtr, std::string> hidden_act = "gelu_pytorch_tahn";
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
    // size_t n = weight->nelements();
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

    auto inv_freq = Tensor::create(ctx, {dim / 2}, DType::FLOAT32);
    for (int i = 0; i < dim / 2; ++i) {
      float exponent = (2.0f * (float)i) / (float)dim;
      float val = 1.0f / std::pow(base, exponent);
      inv_freq->set_from_float_flat((size_t)i, val);
    }
    float attention_factor = 1.0f; // Unused for default RoPE
    return {inv_freq, attention_factor};
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
    if (std::holds_alternative<std::string>(cfg.hidden_act)) {
      lc.hidden_act = std::get<std::string>(cfg.hidden_act);
    } else {
      // 默认回退至常用激活函数，保证与 LlamaConfig 类型一致
      lc.hidden_act = "silu";
    }
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
      // Python: position_ids[None, ...].expand(3, position_ids.shape[0], -1)
      // C++: unsqueeze(0) -> expand({3, B, S})
      int B = position_ids->shape[0];
      int S = position_ids->shape[1];
      auto pos_unsqueezed = unsqueeze(position_ids, 0); // [1, B, S]
      pos3 = pos_unsqueezed->expand({3, B, S});         // [3, B, S]
    } else if (position_ids->shape.size() == 3 && position_ids->shape[0] == 3) {
      pos3 = position_ids;
    } else {
      throw std::runtime_error("Qwen3VLTextRotaryEmbedding::forward: "
                               "position_ids must be [B,S] or [3,B,S]");
    }

    int B = pos3->shape[1];
    int S = pos3->shape[2];
    int half_dim = (int)inv_freq->shape[0];

    // Python equivalent:
    // inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3,
    // position_ids.shape[1], -1) position_ids_expanded = position_ids[:, :,
    // None, :].float() # shape(3, bs, 1, positions) freqs =
    // (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2,
    // 3)

    // inv_freq: [half_dim] -> [1, 1, half_dim, 1] -> [3, B, half_dim, 1]
    auto inv_freq_4d = inv_freq->reshape_view({1, 1, half_dim, 1});
    auto inv_freq_expanded = inv_freq_4d->expand({3, B, half_dim, 1});

    // pos3: [3, B, S] -> [3, B, 1, S]
    auto pos3_4d = pos3->reshape_view({3, B, 1, S});

    // Matrix multiplication: [3, B, half_dim, 1] @ [3, B, 1, S] -> [3, B,
    // half_dim, S]
    auto freqs_raw = Tensor::matmul_cache_friendly(inv_freq_expanded, pos3_4d);

    // Transpose last two dimensions: [3, B, half_dim, S] -> [3, B, S, half_dim]
    auto freqs = freqs_raw->transpose_view(-2, -1);

    auto freqs_t = apply_interleaved_mrope(freqs, mrope_section);

    auto emb = concat({freqs_t, freqs_t}, -1);

    auto cos = emb->tensor_cos(emb);
    auto sin = emb->tensor_sin(emb);

    // Apply attention scaling
    if (attention_scaling != 1.0f) {
      cos = cos->elementwise_unary(
          cos, [this](float v) { return v * this->attention_scaling; });
      sin = sin->elementwise_unary(
          sin, [this](float v) { return v * this->attention_scaling; });
    }

    return {cos, sin};
  }
};

struct Qwen3RMSNorm {
  TensorPtr weight;
  float variance_epsilon;
  Qwen3RMSNorm(const std::shared_ptr<Context> &ctx, int hidden_size,
               float eps = 1e-6) {
    if (!ctx)
      throw std::runtime_error("Qwen3RMSNorm: ctx expired");
    weight = Tensor::ones(ctx, {hidden_size}, DType::FLOAT32);
    variance_epsilon = eps;
  }
  TensorPtr forward(const TensorPtr &hidden_states) {
    // 参数校验
    if (!hidden_states || !weight)
      throw std::runtime_error("Qwen3RMSNorm: null input or weight");
    auto ctx = hidden_states->ctx.lock();
    if (!ctx)
      throw std::runtime_error("Qwen3RMSNorm: ctx expired");
    auto shape = hidden_states->shape;
    if (shape.empty())
      throw std::runtime_error("Qwen3RMSNorm: input rank must be >= 1");

    // 获取最后一维作为 hidden_size，其余维度合并为 batch_seq
    int hidden_size = shape.back();
    size_t batch_seq = hidden_states->nelements() / hidden_size;
    auto in_dtype = hidden_states->dtype;

    // 创建输出张量，统一使用 float32 进行计算
    auto Y = Tensor::create(ctx, shape, DType::FLOAT32);

    // 遍历每个 token（或 patch）向量
    for (size_t i = 0; i < batch_seq; ++i) {
      // 1. 计算该向量元素的平方均值（方差）
      float variance = 0.0f;
      for (int j = 0; j < hidden_size; ++j) {
        float v = hidden_states->get_as_float_flat(i * hidden_size + j);
        variance += v * v;
      }
      variance /= (float)hidden_size;

      // 2. 计算 rsqrt(variance + epsilon)，与Python实现一致
      float rsqrt_var = 1.0f / std::sqrt(variance + variance_epsilon);

      // 3. 归一化后乘以可学习权重，写入输出
      for (int j = 0; j < hidden_size; ++j) {
        float v = hidden_states->get_as_float_flat(i * hidden_size + j);
        float normalized = v * rsqrt_var;
        float w = weight->get_as_float_flat(j);
        Y->set_from_float_flat(i * hidden_size + j, normalized * w);
      }
    }

    // 如果原始 dtype 不是 float32，则转换回去
    if (in_dtype != DType::FLOAT32) {
      return Y->astype(in_dtype);
    }
    return Y;
  }

  std::string extre_repe() const {
    return std::string("(") + std::to_string(weight ? weight->shape[0] : 0) +
           ")" + ", eps=" + std::to_string(variance_epsilon);
  }
};

struct PastKeyValueCache {
  std::shared_ptr<Context> ctx;
  int num_layers;
  int num_kv_heads;
  int head_dim;
  int max_seq_len;
  int batch_size;
  std::vector<TensorPtr> k_caches; // per-layer: [B, L_max, Kvh, D]
  std::vector<TensorPtr> v_caches; // per-layer: [B, L_max, Kvh, D]
  std::vector<int>
      cache_len_per_batch; // assume uniform length across batch for now

  PastKeyValueCache(const std::shared_ptr<Context> &ctx, int num_layers,
                    int num_kv_heads, int head_dim, int max_seq_len)
      : ctx(ctx), num_layers(num_layers), num_kv_heads(num_kv_heads),
        head_dim(head_dim), max_seq_len(max_seq_len), batch_size(0) {
    if (!ctx)
      throw std::runtime_error("PastKeyValueCache: ctx expired");
    k_caches.resize(num_layers);
    v_caches.resize(num_layers);
  }

  void ensure_initialized(int B) {
    if (batch_size == B && k_caches[0])
      return;
    batch_size = B;
    cache_len_per_batch.assign(B, 0);
    for (int l = 0; l < num_layers; ++l) {
      k_caches[l] = Tensor::create(
          ctx, {B, max_seq_len, num_kv_heads, head_dim}, DType::FLOAT32);
      v_caches[l] = Tensor::create(
          ctx, {B, max_seq_len, num_kv_heads, head_dim}, DType::FLOAT32);
    }
  }

  int active_seq_len() const {
    if (cache_len_per_batch.empty())
      return 0;
    int lmin = cache_len_per_batch[0];
    for (int b = 1; b < batch_size; ++b) {
      lmin = std::min(lmin, cache_len_per_batch[b]);
    }
    return lmin;
  }

  // cache_position-aware update overload for static cache semantics
  std::pair<TensorPtr, TensorPtr> update(const TensorPtr &key_states,
                                         const TensorPtr &value_states,
                                         int layer_idx, int sliding_window,
                                         const TensorPtr &cache_position) {
    if (!key_states || !value_states)
      throw std::runtime_error("PastKeyValueCache.update: null K/V");
    auto key_ctx = key_states->ctx.lock();
    if (!key_ctx)
      throw std::runtime_error("PastKeyValueCache.update: ctx expired");
    int B = key_states->shape[0];
    int Kvh = key_states->shape[1];
    int S_new = key_states->shape[2];
    int D = key_states->shape[3];
    ensure_initialized(B);

    bool has_static_pos = (cache_position != nullptr);

    // Append or static write per batch
    for (int b = 0; b < B; ++b) {
      int prev_len = cache_len_per_batch[b];
      int total_len = prev_len + S_new;
      int max_written_pos = prev_len - 1; // track max position written

      for (int j = 0; j < S_new; ++j) {
        int dst_seq;
        if (has_static_pos) {
          // cache_position can be shape [B, S_new] or [S_new]; interpret as
          // absolute positions
          int pos_index;
          if ((int)cache_position->shape.size() == 2) {
            // [B, S_new]
            pos_index = (int)cache_position->get_as_float_flat(
                ((size_t)b * (size_t)cache_position->shape[1]) + (size_t)j);
          } else if ((int)cache_position->shape.size() == 1) {
            // [S_new]
            pos_index = (int)cache_position->get_as_float_flat((size_t)j);
          } else {
            throw std::runtime_error(
                "PastKeyValueCache.update: cache_position must be 1D or 2D");
          }
          dst_seq = std::max(0, std::min(pos_index, max_seq_len - 1));
        } else {
          dst_seq = prev_len + j;
        }
        if (dst_seq >= max_seq_len)
          break; // overflow guard
        max_written_pos = std::max(max_written_pos, dst_seq);

        for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
          for (int d = 0; d < head_dim; ++d) {
            // src index in key_states: [B, Kvh, S_new, D]
            size_t k_src = ((size_t)b * (size_t)num_kv_heads * (size_t)S_new *
                            (size_t)head_dim) +
                           ((size_t)kvh * (size_t)S_new * (size_t)head_dim) +
                           ((size_t)j * (size_t)head_dim) + (size_t)d;
            float k_val = key_states->get_as_float_flat(k_src);
            // dst index in cache: [B, L_max, Kvh, D]
            size_t k_dst =
                ((size_t)b * (size_t)max_seq_len * (size_t)num_kv_heads *
                 (size_t)head_dim) +
                ((size_t)dst_seq * (size_t)num_kv_heads * (size_t)head_dim) +
                ((size_t)kvh * (size_t)head_dim) + (size_t)d;
            k_caches[layer_idx]->set_from_float_flat(k_dst, k_val);

            size_t v_src = ((size_t)b * (size_t)num_kv_heads * (size_t)S_new *
                            (size_t)head_dim) +
                           ((size_t)kvh * (size_t)S_new * (size_t)head_dim) +
                           ((size_t)j * (size_t)head_dim) + (size_t)d;
            float v_val = value_states->get_as_float_flat(v_src);
            size_t v_dst =
                ((size_t)b * (size_t)max_seq_len * (size_t)num_kv_heads *
                 (size_t)head_dim) +
                ((size_t)dst_seq * (size_t)num_kv_heads * (size_t)head_dim) +
                ((size_t)kvh * (size_t)head_dim) + (size_t)d;
            v_caches[layer_idx]->set_from_float_flat(v_dst, v_val);
          }
        }
      }

      // Update active length: for static cache use max_written_pos+1; for
      // dynamic append use min(total_len, max_seq_len)
      if (has_static_pos)
        cache_len_per_batch[b] = std::min(max_written_pos + 1, max_seq_len);
      else
        cache_len_per_batch[b] = std::min(total_len, max_seq_len);
    }

    // Sliding window compaction only in dynamic mode
    if (!has_static_pos && sliding_window > 0) {
      for (int b = 0; b < B; ++b) {
        int L = cache_len_per_batch[b];
        if (L > sliding_window) {
          int src_start = L - sliding_window;
          for (int j = 0; j < sliding_window; ++j) {
            int dst_seq = j;
            int src_seq = src_start + j;
            for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
              for (int d = 0; d < head_dim; ++d) {
                size_t k_src = ((size_t)b * (size_t)max_seq_len *
                                (size_t)num_kv_heads * (size_t)head_dim) +
                               ((size_t)src_seq * (size_t)num_kv_heads *
                                (size_t)head_dim) +
                               ((size_t)kvh * (size_t)head_dim) + (size_t)d;
                float k_val = k_caches[layer_idx]->get_as_float_flat(k_src);
                size_t k_dst = ((size_t)b * (size_t)max_seq_len *
                                (size_t)num_kv_heads * (size_t)head_dim) +
                               ((size_t)dst_seq * (size_t)num_kv_heads *
                                (size_t)head_dim) +
                               ((size_t)kvh * (size_t)head_dim) + (size_t)d;
                k_caches[layer_idx]->set_from_float_flat(k_dst, k_val);

                size_t v_src = ((size_t)b * (size_t)max_seq_len *
                                (size_t)num_kv_heads * (size_t)head_dim) +
                               ((size_t)src_seq * (size_t)num_kv_heads *
                                (size_t)head_dim) +
                               ((size_t)kvh * (size_t)head_dim) + (size_t)d;
                float v_val = v_caches[layer_idx]->get_as_float_flat(v_src);
                size_t v_dst = ((size_t)b * (size_t)max_seq_len *
                                (size_t)num_kv_heads * (size_t)head_dim) +
                               ((size_t)dst_seq * (size_t)num_kv_heads *
                                (size_t)head_dim) +
                               ((size_t)kvh * (size_t)head_dim) + (size_t)d;
                v_caches[layer_idx]->set_from_float_flat(v_dst, v_val);
              }
            }
          }
          cache_len_per_batch[b] = sliding_window;
        }
      }
    }

    // Build output slices: [B, Kvh, L_active, D]
    int L_active = active_seq_len();
    auto k_slice = k_caches[layer_idx]->slice_view_step(1, 0, L_active, 1);
    auto v_slice = v_caches[layer_idx]->slice_view_step(1, 0, L_active, 1);
    auto k_out_pv = permute_view(k_slice, {0, 2, 1, 3});
    auto v_out_pv = permute_view(v_slice, {0, 2, 1, 3});
    // Return contiguous k/v tensors
    auto k_out = Tensor::create(ctx, k_out_pv->shape, k_out_pv->dtype);
    k_out->assign_from(k_out_pv);
    auto v_out = Tensor::create(ctx, v_out_pv->shape, v_out_pv->dtype);
    v_out->assign_from(v_out_pv);
    return {k_out, v_out};
  }
};

struct Qwen3Attention {
  /// Multi-headed attention from 'Attention Is All You Need' paper

  // Configuration and layer info
  std::shared_ptr<Context> ctx;
  Qwen3VLTextConfig config;
  int layer_idx;
  std::optional<std::string> layer_type;

  // Attention parameters
  int head_dim;
  int num_key_value_groups;
  float scaling;
  double attention_dropout;
  bool is_causal;
  std::optional<int> sliding_window;

  // Linear projection layers (weights)
  TensorPtr q_proj_weight;
  TensorPtr k_proj_weight;
  TensorPtr v_proj_weight;
  TensorPtr o_proj_weight;

  // Optional bias tensors
  TensorPtr q_proj_bias;
  TensorPtr k_proj_bias;
  TensorPtr v_proj_bias;
  TensorPtr o_proj_bias;

  // RMS normalization layers
  std::unique_ptr<Qwen3RMSNorm> q_norm;
  std::unique_ptr<Qwen3RMSNorm> k_norm;

  // Legacy single-batch KV cache state (per layer)
  mutable TensorPtr k_cache;
  mutable TensorPtr v_cache;
  mutable int cache_len = 0;
  mutable int max_seq_len_cached = 0;

  Qwen3Attention(const std::shared_ptr<Context> &ctx,
                 const Qwen3VLTextConfig &config, int layer_idx)
      : ctx(ctx), config(config), layer_idx(layer_idx) {

    if (!ctx)
      throw std::runtime_error("Qwen3Attention: ctx expired");

    // Initialize layer type
    if (config.layer_types.has_value() &&
        layer_idx < config.layer_types->size()) {
      layer_type = (*config.layer_types)[layer_idx];
    }

    // head_dim from config or derive
    head_dim = config.head_dim > 0
                   ? config.head_dim
                   : config.hidden_size / config.num_attention_heads;

    // key/value groups
    num_key_value_groups =
        config.num_attention_heads / config.num_key_value_heads;

    // scaling
    scaling = std::pow(head_dim, -0.5f);

    attention_dropout = config.attention_dropout;
    // 从配置读取是否因果掩码
    is_causal = config.is_causal;

    // sliding window if layer type is sliding_attention
    if (layer_type.has_value() && layer_type.value() == "sliding_attention") {
      sliding_window = config.sliding_window;
    } else {
      sliding_window = std::nullopt;
    }

    // Initialize projection weights
    q_proj_weight = Tensor::create(
        ctx, {config.hidden_size, config.num_attention_heads * head_dim},
        DType::FLOAT32);
    k_proj_weight = Tensor::create(
        ctx, {config.hidden_size, config.num_key_value_heads * head_dim},
        DType::FLOAT32);
    v_proj_weight = Tensor::create(
        ctx, {config.hidden_size, config.num_key_value_heads * head_dim},
        DType::FLOAT32);
    o_proj_weight = Tensor::create(
        ctx, {config.num_attention_heads * head_dim, config.hidden_size},
        DType::FLOAT32);

    // Optional biases
    if (config.attention_bias) {
      q_proj_bias = Tensor::create(ctx, {config.num_attention_heads * head_dim},
                                   DType::FLOAT32);
      k_proj_bias = Tensor::create(ctx, {config.num_key_value_heads * head_dim},
                                   DType::FLOAT32);
      v_proj_bias = Tensor::create(ctx, {config.num_key_value_heads * head_dim},
                                   DType::FLOAT32);
      o_proj_bias = Tensor::create(ctx, {config.hidden_size}, DType::FLOAT32);
    }

    // RMSNorm layers（受配置开关控制）
    if (config.use_qk_rmsnorm) {
      q_norm = std::make_unique<Qwen3RMSNorm>(ctx, head_dim, config.rms_norm_eps);
      k_norm = std::make_unique<Qwen3RMSNorm>(ctx, head_dim, config.rms_norm_eps);
    }

    // KV cache capacity
    max_seq_len_cached = config.max_position_embeddings;
  }

  // Linear projection helpers
  TensorPtr q_proj(const TensorPtr &x) const {
    if (config.attention_bias && q_proj_bias) {
      return Tensor::linear(x, q_proj_weight, q_proj_bias);
    } else {
      return Tensor::linear(x, q_proj_weight);
    }
  }

  TensorPtr k_proj(const TensorPtr &x) const {
    if (config.attention_bias && k_proj_bias) {
      return Tensor::linear(x, k_proj_weight, k_proj_bias);
    } else {
      return Tensor::linear(x, k_proj_weight);
    }
  }

  TensorPtr v_proj(const TensorPtr &x) const {
    if (config.attention_bias && v_proj_bias) {
      return Tensor::linear(x, v_proj_weight, v_proj_bias);
    } else {
      return Tensor::linear(x, v_proj_weight);
    }
  }

  TensorPtr o_proj(const TensorPtr &x) const {
    if (config.attention_bias && o_proj_bias) {
      return Tensor::linear(x, o_proj_weight, o_proj_bias);
    } else {
      return Tensor::linear(x, o_proj_weight);
    }
  }

  // Legacy single-batch cache init
  void init_cache(int max_len) const {
    if (!ctx)
      throw std::runtime_error("Qwen3Attention::init_cache: ctx expired");
    k_cache = Tensor::create(
        ctx, {max_len, config.num_key_value_heads, head_dim}, DType::FLOAT32);
    v_cache = Tensor::create(
        ctx, {max_len, config.num_key_value_heads, head_dim}, DType::FLOAT32);
    cache_len = 0;
  }

  // Convenience overloads
  TensorPtr forward(const TensorPtr &hidden_states, const TensorPtr &cos,
                    const TensorPtr &sin) const {
    return forward(hidden_states, cos, sin, nullptr, nullptr);
  }

  TensorPtr forward(const TensorPtr &hidden_states, const TensorPtr &cos,
                    const TensorPtr &sin,
                    const TensorPtr &attention_mask) const {
    return forward(hidden_states, cos, sin, attention_mask, nullptr);
  }

  // Cache-aware attention main path
  TensorPtr
  forward(const TensorPtr &hidden_states, const TensorPtr &cos,
          const TensorPtr &sin, const TensorPtr &attention_mask,
          const std::shared_ptr<PastKeyValueCache> &past_key_values) const {
    if (!hidden_states || !cos || !sin)
      throw std::runtime_error("Qwen3Attention::forward: null input");
    auto ctx_local = hidden_states->ctx.lock();
    if (!ctx_local)
      throw std::runtime_error("Qwen3Attention::forward: ctx expired");

    int rank = (int)hidden_states->shape.size();
    if (rank != 2 && rank != 3)
      throw std::runtime_error(
          "Qwen3Attention::forward: hidden_states must be [S,H] or [B,S,H]");

    int B = (rank == 3) ? hidden_states->shape[0] : 1;
    int S = (rank == 3) ? hidden_states->shape[1] : hidden_states->shape[0];
    int Hsz = (rank == 3) ? hidden_states->shape[2] : hidden_states->shape[1];

    // Flatten to [B*S, H]
    int BS = B * S;
    TensorPtr hs2d = hidden_states->is_contiguous_row_major()
                         ? hidden_states->reshape_view({BS, Hsz})
                         : hidden_states->copy()->reshape_view({BS, Hsz});

    // Q/K/V projections
    auto q_all = q_proj(hs2d);
    auto k_all = k_proj(hs2d);
    auto v_all = v_proj(hs2d);

    int num_heads = config.num_attention_heads;
    int num_kv_heads = config.num_key_value_heads;

    // Reshape to [B,S,heads,D] first, then apply RMSNorm on each head
    auto q_4d = q_all->reshape_view({B, S, num_heads, head_dim});
    auto k_4d = k_all->reshape_view({B, S, num_kv_heads, head_dim});
    auto v_4d = v_all->reshape_view({B, S, num_kv_heads, head_dim});

    // Apply RMSNorm to Q and K after reshaping (matching Python implementation)
    if (q_norm && q_norm->weight) {
      // Apply RMSNorm on the last dimension (head_dim) for each head
      q_4d = q_norm->forward(q_4d);
    }
    if (k_norm && k_norm->weight) {
      // Apply RMSNorm on the last dimension (head_dim) for each head
      k_4d = k_norm->forward(k_4d);
    }

    auto q = permute_view(q_4d, {0, 2, 1, 3});
    auto k = permute_view(k_4d, {0, 2, 1, 3});
    auto v = permute_view(v_4d, {0, 2, 1, 3});

    // Ensure q/k/v are contiguous after permutation
    if (!q->is_contiguous_row_major()) {
      auto q_cont = Tensor::create(ctx_local, q->shape, q->dtype);
      q_cont->assign_from(q);
      q = q_cont;
    }
    if (!k->is_contiguous_row_major()) {
      auto k_cont = Tensor::create(ctx_local, k->shape, k->dtype);
      k_cont->assign_from(k);
      k = k_cont;
    }
    if (!v->is_contiguous_row_major()) {
      auto v_cont = Tensor::create(ctx_local, v->shape, v->dtype);
      v_cont->assign_from(v);
      v = v_cont;
    }
    float scale = scaling;
    q = q->elementwise_unary(q, [scale](float v) { return v * scale; });

    // RoPE
    auto qk_rot = ow::nn::apply_rotary_pos_emb(q, k, cos, sin, nullptr,
                                               /*unsqueeze_dim=*/1);
    q = std::get<0>(qk_rot);
    k = std::get<1>(qk_rot);

    // Use external cache if provided
    TensorPtr k_active;
    TensorPtr v_active;
    int L_keys = S;
    int L_prev = 0;
    if (past_key_values) {
      auto kv = past_key_values->update(k, v, layer_idx,
                                        sliding_window.value_or(-1), nullptr);
      k_active = kv.first;
      v_active = kv.second;
      L_keys = past_key_values->active_seq_len();
      L_prev = std::max(0, L_keys - S);
    } else {
      k_active = k;
      v_active = v;
    }

    auto attn_out =
        Tensor::create(ctx_local, {B, num_heads, S, head_dim}, DType::FLOAT32);

    // Compute per batch/head attention
    for (int b = 0; b < B; ++b) {
      for (int hidx = 0; hidx < num_heads; ++hidx) {
        int group_size = std::max(1, num_heads / num_kv_heads);
        int kv_h = hidx / group_size;
        auto q_bh = q->select_view(0, b)->select_view(0, hidx);
        auto k_bkv = k_active->select_view(0, b)->select_view(0, kv_h);
        auto k_T = k_bkv->transpose_view(0, 1);
        // Ensure k_T is contiguous for matmul
        if (!k_T->is_contiguous_row_major()) {
          auto k_T_cont = Tensor::create(ctx_local, k_T->shape, k_T->dtype);
          k_T_cont->assign_from(k_T);
          k_T = k_T_cont;
        }
        
        auto scores = Tensor::matmul_cache_friendly(q_bh, k_T); // [S, L_keys]

        // Causal mask（仅当启用因果掩码时）
        const float NEG_INF = -1e9f;
        if (is_causal) {
          auto causal_mask =
              Tensor::create(ctx_local, {S, L_keys}, DType::FLOAT32);
          for (int i = 0; i < S; ++i) {
            int i_abs = L_prev + i;
            for (int j = 0; j < L_keys; ++j) {
              float m = (j > i_abs) ? NEG_INF : 0.0f;
              causal_mask->set_from_float_flat((size_t)i * L_keys + j, m);
            }
          }
          scores = scores->tensor_add(scores, causal_mask);
        }

        // Attention mask broadcastable to [S, L_keys]
        if (attention_mask) {
          if (attention_mask->shape.size() == 4) {
            auto am2d = attention_mask->select_view(0, b)->select_view(0, 0);
            scores = scores->tensor_add(scores, am2d);
          } else if (attention_mask->shape.size() == 3) {
            auto kv_mask = attention_mask->select_view(0, b)->select_view(0, 0);
            auto kv2d = kv_mask->reshape_view({1, L_keys})->repeat({S, 1});
            scores = scores->tensor_add(scores, kv2d);
          } else if (attention_mask->shape.size() == 2) {
            auto kv_mask = attention_mask->select_view(0, b);
            auto kv2d = kv_mask->reshape_view({1, L_keys})->repeat({S, 1});
            scores = scores->tensor_add(scores, kv2d);
          }
        }

        auto weights = ow::nn::softmax(scores, -1);
        auto v_bkv = v_active->select_view(0, b)->select_view(0, kv_h);
        auto context = Tensor::matmul_cache_friendly(weights, v_bkv);

        for (int i = 0; i < S; ++i) {
          for (int d = 0; d < head_dim; ++d) {
            float val = context->get_as_float_flat((size_t)i * head_dim + d);
            size_t dst = ((size_t)b * num_heads * S * head_dim) +
                         ((size_t)hidx * S * head_dim) +
                         ((size_t)i * head_dim) + d;
            attn_out->set_from_float_flat(dst, val);
          }
        }
      }
    }

    auto attn_perm = permute_view(attn_out, {0, 2, 1, 3});
    // Materialize attn_perm as contiguous before reshape
    auto attn_perm_cont = Tensor::create(ctx_local, attn_perm->shape, attn_perm->dtype);
    attn_perm_cont->assign_from(attn_perm);
    auto attn_flat = attn_perm_cont->reshape_view({BS, num_heads * head_dim});
    auto out = o_proj(attn_flat);
    if (rank == 3) {
      return out->reshape_view({B, S, Hsz});
    } else {
      return out->reshape_view({S, Hsz});
    }
  }
};

#ifdef OW_NN_ENABLE_QWEN3MLP
struct Qwen3MLP {
  int hidden_size = 0;
  int intermediate_size = 0;
  TensorPtr gate_proj;
  TensorPtr up_proj;
  TensorPtr down_proj;
  std::variant<TensorPtr, std::string> act_fn;

  Qwen3MLP(const std::shared_ptr<Context> &ctx,
           const Qwen3VLTextConfig &config) {
    if (!ctx)
      throw std::runtime_error("Qwen3MLP: ctx expired");
    // 原因：如果 config.hidden_size 不大于 0，说明配置非法，抛出异常以终止构造
    hidden_size = config.hidden_size > 0
                      ? config.hidden_size
                      : throw std::runtime_error(
                            "Qwen3MLP: hidden_size must be positive");
    intermediate_size =
        config.intermediate_size > 0
            ? config.intermediate_size
            : throw std::runtime_error(
                  "Qwen3MLP: intermediate_size must be positive");

    // // config.hidden_act 是 std::variant<TensorPtr,
    // std::string>，需要判断类型 if
    // (std::holds_alternative<std::string>(config.hidden_act)) {
    //   act_fn = std::get<std::string>(config.hidden_act);
    // } else {
    //   auto ptr = std::get<TensorPtr>(config.hidden_act);
    //   if (ptr) {
    //     act_fn = ptr;
    //   } else {
    //     // 如果 TensorPtr 为空，则退回到默认字符串
    //     act_fn = std::string("silu");
    //   }
    // }
    act_fn = config.hidden_act;

    gate_proj = Tensor::linear(gate_proj, hidden_size, intermediate_size, false)
                    ->astype(DType::FLOAT32);
    up_proj = Tensor::linear(up_proj, hidden_size, intermediate_size, false)
                  ->astype(DType::FLOAT32);
    down_proj = Tensor::linear(down_proj, intermediate_size, hidden_size, false)
                    ->astype(DType::FLOAT32);
  }
  TensorPtr forward(const TensorPtr &x) {
    if (std::holds_alternative<std::string>(act_fn)) {
      const auto &act_str = std::get<std::string>(act_fn);
      if (!act_str.empty()) {
        // 根据字符串名称应用激活函数并继续后续计算
        auto gate = (*this->gate_proj)(x);
        auto up = (*this->up_proj)(x);
        auto activated = apply_activation(gate, act_str);
        down_proj = down_proj->tensor_mul(activated, up);
      }
    } else if (std::holds_alternative<TensorPtr>(act_fn)) {
      const auto &act_tensor = std::get<TensorPtr>(act_fn);
      if (act_tensor) {
        // 在这里这接选择用silu激活
        down_proj = down_proj->tensor_mul(silu((*this->gate_proj)(x)),
                                          (*this->up_proj)(x));
      }
    }
    return down_proj;
  }
};
#endif // OW_NN_ENABLE_QWEN3MLP

} // namespace ow::nn