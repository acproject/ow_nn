#pragma once
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include "tensor.hpp"
#include "context.hpp"

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
};

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
  
    std::optional<std::variant<RopeParameters, std::unordered_map<std::string, RopeParameters>>> rope_parameters = std::nullopt;
    bool attention_bias = false;
    std::optional<std::vector<std::string>> layer_types = std::nullopt;
    double attention_dropout = 0.0;

    std::string model_type = "qwen3_vl";
    std::string base_config_key = "vision_config";

    Qwen3VLTextConfig() = default;
};

struct Qwen3VLVisionConfig {
  static inline const  std::string model_type = "qwen3_vl";
  static inline const   std::string base_config_key = "vision_config";
    int depth=27;
    int hidden_size=1152;
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
    static inline const std::unordered_map<std::string, std::string> sub_config = {
        {"vision_config", "Qwen3VLVisionConfig"},
        {"text_config", "Qwen3VLTextConfig"}
    };
    static inline const std::vector<std::string> keys_to_ignore_at_inference = {"past_key_values"};

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
    TensorPtr weight;           // shape: [hidden_size]
    float variance_epsilon = 1e-6f;

    Qwen2RMSNorm(int hidden_size, float eps, const std::shared_ptr<Context>& ctx)
        : variance_epsilon(eps) {
        if (!ctx) throw std::runtime_error("Qwen2RMSNorm: ctx expired");
        weight = Tensor::create(ctx, {hidden_size}, DType::FLOAT32);
        size_t n = weight->nelements();
        for (size_t i = 0; i < n; ++i) weight->set_from_float_flat(i, 1.0f);
    }

    TensorPtr forward(const TensorPtr& hidden_states) const {
        if (!hidden_states || !weight) throw std::runtime_error("Qwen2RMSNorm: null input or weight");
        auto ctx = hidden_states->ctx.lock();
        if (!ctx) throw std::runtime_error("Qwen2RMSNorm: ctx expired");
        auto shape = hidden_states->shape;
        if (shape.empty()) throw std::runtime_error("Qwen2RMSNorm: input rank must be >= 1");
        int hidden_size = shape.back();
        size_t batch_seq = hidden_states->nelements() / hidden_size;
        auto in_dtype = hidden_states->dtype;
        auto Y = Tensor::create(ctx, shape, DType::FLOAT32);
        for (size_t i = 0; i < batch_seq; ++i) {
            float mean_sq = 0.0f;
            for (int j = 0; j < hidden_size; ++j) {
                float v = hidden_states->get_as_float_flat(i * hidden_size + j);
                mean_sq += v * v;
            }
            mean_sq /= (float)hidden_size;
            float inv_rms = 1.0f / std::sqrt(mean_sq + variance_epsilon);
            for (int j = 0; j < hidden_size; ++j) {
                float v = hidden_states->get_as_float_flat(i * hidden_size + j) * inv_rms;
                float w = weight->get_as_float_flat(j);
                Y->set_from_float_flat(i * hidden_size + j, v * w);
            }
        }
        if (in_dtype != DType::FLOAT32) {
            return Y->astype(in_dtype);
        }
        return Y;
    }

    std::string extra_repr() const {
        return std::string("(") + std::to_string(weight ? weight->shape[0] : 0) + ")" + ", eps=" + std::to_string(variance_epsilon);
    }
};

} // namespace ow::nn