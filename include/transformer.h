#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include "common.h"

namespace ow::nn {

// Forward declarations to avoid heavy includes
struct Tensor;
class Context;
using TensorPtr = std::shared_ptr<Tensor>;

// RMSNorm layer
class RMSNorm {
public:
    TensorPtr weight;
    float eps;
    
    RMSNorm(const TensorPtr& weight, float eps = 1e-6f) 
        : weight(weight), eps(eps) {}
    
    TensorPtr forward(const TensorPtr& x);
};

// Rotary positional embedding
class RotaryEmbedding {
public:
    int dim;
    int max_seq_len;
    float base;
    TensorPtr cos_cached;
    TensorPtr sin_cached;
    
    RotaryEmbedding(int dim, int max_seq_len = 2048, float base = 10000.0f);
    
    std::pair<TensorPtr, TensorPtr> forward(int seq_len, const std::shared_ptr<Context>& ctx);
    TensorPtr apply_rotary_pos_emb(const TensorPtr& x, const TensorPtr& cos, const TensorPtr& sin);
};

// Multi-Head Attention
class MultiHeadAttention {
public:
    TensorPtr q_proj;
    TensorPtr k_proj; 
    TensorPtr v_proj;
    TensorPtr o_proj;
    std::shared_ptr<RMSNorm> q_norm;
    std::shared_ptr<RMSNorm> k_norm;
    std::shared_ptr<RotaryEmbedding> rotary_emb;
    
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    
    MultiHeadAttention(const TensorPtr& q_proj, const TensorPtr& k_proj, 
                      const TensorPtr& v_proj, const TensorPtr& o_proj,
                      const TensorPtr& q_norm_weight, const TensorPtr& k_norm_weight,
                      int num_heads, int num_kv_heads, int hidden_size);
    
    TensorPtr forward(const TensorPtr& hidden_states, int seq_len);
};

// MoE Expert
class Expert {
public:
    TensorPtr gate_proj;
    TensorPtr up_proj;
    TensorPtr down_proj;
    
    Expert(const TensorPtr& gate_proj, const TensorPtr& up_proj, const TensorPtr& down_proj)
        : gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj) {}
    
    TensorPtr forward(const TensorPtr& x);
};

// Sparse MoE Block
class SparseMoEBlock {
public:
    TensorPtr gate;  // Router
    std::vector<std::shared_ptr<Expert>> experts;
    int num_experts;
    int top_k;
    
    SparseMoEBlock(const TensorPtr& gate, const std::vector<std::shared_ptr<Expert>>& experts, 
                   int top_k = 8) 
        : gate(gate), experts(experts), num_experts(experts.size()), top_k(top_k) {}
    
    TensorPtr forward(const TensorPtr& x);
};

// Decoder Layer
class DecoderLayer {
public:
    std::shared_ptr<MultiHeadAttention> self_attn;
    std::shared_ptr<SparseMoEBlock> mlp;
    std::shared_ptr<RMSNorm> input_layernorm;
    std::shared_ptr<RMSNorm> post_attention_layernorm;
    
    DecoderLayer(std::shared_ptr<MultiHeadAttention> self_attn,
                std::shared_ptr<SparseMoEBlock> mlp,
                std::shared_ptr<RMSNorm> input_layernorm,
                std::shared_ptr<RMSNorm> post_attention_layernorm)
        : self_attn(self_attn), mlp(mlp), 
          input_layernorm(input_layernorm), post_attention_layernorm(post_attention_layernorm) {}
    
    TensorPtr forward(const TensorPtr& hidden_states, int seq_len);
};

// Qwen3VL Text Model
class Qwen3VLTextModel {
public:
    TensorPtr embed_tokens;
    std::vector<std::shared_ptr<DecoderLayer>> layers;
    std::shared_ptr<RMSNorm> norm;
    std::shared_ptr<RotaryEmbedding> rotary_emb;
    
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    
    Qwen3VLTextModel(const TensorPtr& embed_tokens,
                    const std::vector<std::shared_ptr<DecoderLayer>>& layers,
                    std::shared_ptr<RMSNorm> norm,
                    int vocab_size, int hidden_size, int num_layers, 
                    int num_heads, int num_kv_heads);
    
    TensorPtr forward(const std::vector<int>& input_ids);
};

// Lightweight WeightLoader interface
class WeightLoader {
private:
    std::unordered_map<std::string, TensorPtr> weights;

public:
    void add_weight(const std::string& name, const TensorPtr& weight);
    TensorPtr get_weight(const std::string& name) const;
    std::shared_ptr<Qwen3VLTextModel> build_model();
};

} // namespace ow::nn