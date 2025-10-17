#include "../include/transformer.h"
#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace ow::nn {

// Ensure a 2D linear weight has expected dims, but avoid copying.
// If loaded as [out_dim, in_dim], leave it as-is; matmul handles transposed B.
static TensorPtr ensure_linear_weight_or_transpose(const TensorPtr& W, int expected_in_dim, int expected_out_dim = -1) {
    if (!W) throw std::runtime_error("ensure_linear_weight_or_transpose: null weight");
    if (W->shape.size() != 2) return W; // Only handle 2D weights
    int r0 = W->shape[0];
    int r1 = W->shape[1];
    bool ok_no_transpose = (r0 == expected_in_dim) && (expected_out_dim < 0 || r1 == expected_out_dim);
    if (ok_no_transpose) return W;
    bool can_transpose = (r1 == expected_in_dim) && (expected_out_dim < 0 || r0 == expected_out_dim);
    if (can_transpose) {
        std::cerr << "[Weight] Using transposed orientation without copy: expected ["
                  << expected_in_dim << "," << (expected_out_dim<0? r0: expected_out_dim)
                  << "] got [" << r0 << "," << r1 << "]" << std::endl;
        return W; // rely on matmul auto-handling
    }
    // Dimensions don't match expectations; return as-is and let call sites fail loudly.
    return W;
}


// WeightLoader method implementations
void WeightLoader::add_weight(const std::string& name, const TensorPtr& weight) {
    weights[name] = weight;
}

TensorPtr WeightLoader::get_weight(const std::string& name) const {
    auto it = weights.find(name);
    return (it != weights.end()) ? it->second : nullptr;
}

// RMSNorm implementation
TensorPtr RMSNorm::forward(const TensorPtr& x) {
    if (!x || !weight) {
        throw std::runtime_error("RMSNorm: null input or weight");
    }
    
    auto ctx = x->ctx.lock();
    if (!ctx) throw std::runtime_error("RMSNorm: ctx expired");
    
    // x shape: [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
    auto shape = x->shape;
    int hidden_size = shape.back();
    size_t batch_seq = x->nelements() / hidden_size;
    
    auto output = Tensor::create(ctx, shape, DType::FLOAT32);
    
    for (size_t i = 0; i < batch_seq; ++i) {
        // Calculate RMS
        float sum_sq = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            float val = x->get_as_float_flat(i * hidden_size + j);
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / hidden_size + eps);
        
        // Normalize and scale
        for (int j = 0; j < hidden_size; ++j) {
            float val = x->get_as_float_flat(i * hidden_size + j);
            float weight_val = weight->get_as_float_flat(j);
            float normalized = (val / rms) * weight_val;
            output->set_from_float_flat(i * hidden_size + j, normalized);
        }
    }
    
    return output;
}

// RotaryEmbedding implementation (declarations are in transformer.h)
// Add missing RotaryEmbedding constructor and forward
RotaryEmbedding::RotaryEmbedding(int dim, int max_seq_len, float base)
    : dim(dim), max_seq_len(max_seq_len), base(base), cos_cached(nullptr), sin_cached(nullptr) {}

std::pair<TensorPtr, TensorPtr> RotaryEmbedding::forward(int seq_len, const std::shared_ptr<Context>& ctx) {
    if (!ctx) throw std::runtime_error("RotaryEmbedding::forward: ctx expired");
    auto cos_vals = Tensor::create(ctx, {seq_len, dim / 2}, DType::FLOAT32);
    auto sin_vals = Tensor::create(ctx, {seq_len, dim / 2}, DType::FLOAT32);
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < dim / 2; ++i) {
            float inv_freq = 1.0f / std::pow(base, (2.0f * i) / dim);
            float angle = pos * inv_freq;
            cos_vals->set_from_float_flat(pos * (dim / 2) + i, std::cos(angle));
            sin_vals->set_from_float_flat(pos * (dim / 2) + i, std::sin(angle));
        }
    }
    cos_cached = cos_vals;
    sin_cached = sin_vals;
    return {cos_vals, sin_vals};
}
TensorPtr RotaryEmbedding::apply_rotary_pos_emb(const TensorPtr& x, const TensorPtr& cos, const TensorPtr& sin) {
    // x: [seq_len, num_heads, head_dim], head_dim assumed even
    auto ctx = x->ctx.lock();
    if (!ctx) throw std::runtime_error("RoPE: ctx expired");
    int seq_len = x->shape[0];
    int num_heads = x->shape[1];
    int head_dim = x->shape[2];
    if ((head_dim % 2) != 0) throw std::runtime_error("RoPE: head_dim must be even");
    auto output = Tensor::create(ctx, x->shape, DType::FLOAT32);
    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < head_dim; d += 2) {
                size_t idx1 = (size_t)i * num_heads * head_dim + (size_t)h * head_dim + d;
                size_t idx2 = idx1 + 1;
                float x1 = x->get_as_float_flat(idx1);
                float x2 = x->get_as_float_flat(idx2);
                size_t cos_idx = (size_t)i * (head_dim/2) + (size_t)(d/2);
                float cos_val = cos->get_as_float_flat(cos_idx);
                float sin_val = sin->get_as_float_flat(cos_idx);
                
                // Fix debug guard variable name
                if (i == 0 && h == 0 && d == 0) {
                    std::cout << "[RoPE] idx1=" << idx1 << " idx2=" << idx2 << " cos_idx=" << cos_idx 
                              << " x1=" << x1 << " x2=" << x2 << " cos=" << cos_val << " sin=" << sin_val << std::endl;
                }
                
                float new_x1 = x1 * cos_val - x2 * sin_val;
                float new_x2 = x1 * sin_val + x2 * cos_val;
                
                output->set_from_float_flat(idx1, new_x1);
                output->set_from_float_flat(idx2, new_x2);
            }
        }
    }
    
    return output;
}

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention(const TensorPtr& q_proj, const TensorPtr& k_proj, 
                                     const TensorPtr& v_proj, const TensorPtr& o_proj,
                                     const TensorPtr& q_norm_weight, const TensorPtr& k_norm_weight,
                                     int num_heads, int num_kv_heads, int hidden_size)
    : q_proj(q_proj), k_proj(k_proj), v_proj(v_proj), o_proj(o_proj),
      num_heads(num_heads), num_kv_heads(num_kv_heads), hidden_size(hidden_size) {
    
    head_dim = hidden_size / num_heads;
    q_norm = std::make_shared<RMSNorm>(q_norm_weight);
    k_norm = std::make_shared<RMSNorm>(k_norm_weight);
    rotary_emb = std::make_shared<RotaryEmbedding>(head_dim);
}

TensorPtr MultiHeadAttention::forward(const TensorPtr& hidden_states, int seq_len) {
    auto ctx = hidden_states->ctx.lock();
    if (!ctx) throw std::runtime_error("MultiHeadAttention: ctx expired");
    std::cout << "[MHA] Start: seq_len=" << seq_len
              << " hidden_size=" << hidden_size
              << " q_proj=[" << q_proj->shape[0] << "," << q_proj->shape[1] << "]"
              << " k_proj=[" << k_proj->shape[0] << "," << k_proj->shape[1] << "]"
              << " v_proj=[" << v_proj->shape[0] << "," << v_proj->shape[1] << "]"
              << " v_proj.dtype=" << (int)v_proj->dtype
              << std::endl;

    // Print a slice of hidden_states row 0
    int hs_print = std::min(hidden_size, 8);
    std::cout << "[MHA] hidden_states[0,:" << hs_print << "]: ";
    for (int h = 0; h < hs_print; ++h) {
        std::cout << hidden_states->get_as_float_flat(0 * hidden_size + h) << (h+1<hs_print?",":"");
    }
    std::cout << std::endl;
    
    // Print a slice of v_proj first row
    int vp_cols = v_proj->shape[1];
    int vp_print = std::min(vp_cols, 8);
    std::cout << "[MHA] v_proj[0,:" << vp_print << "]: ";
    for (int j = 0; j < vp_print; ++j) {
        std::cout << v_proj->get_as_float_flat(0 * vp_cols + j) << (j+1<vp_print?",":"");
    }
    std::cout << std::endl;
    
    // hidden_states: [seq_len, hidden_size]
    // Project to Q, K, V
    auto q = Tensor::matmul_blocked_mt(hidden_states, q_proj);
    auto k = Tensor::matmul_blocked_mt(hidden_states, k_proj);
    // Use scalar fallback for v to validate kernel correctness on narrow N
    auto v = Tensor::matmul_blocked_mt(hidden_states, v_proj, /*block_m*/64, /*block_n*/1, /*block_k*/64, /*nthreads*/1);

    // Print raw q/k/v of position 0 before reshape
    int q_print = std::min(q->shape[1], 8);
    int kv_print = std::min(v->shape[1], 8);
    std::cout << "[MHA] raw q[0,:" << q_print << "]: ";
    for (int d = 0; d < q_print; ++d) {
        std::cout << q->get_as_float_flat(0 * q->shape[1] + d) << (d+1<q_print?",":"");
    }
    std::cout << std::endl;
    std::cout << "[MHA] raw k[0,:" << kv_print << "]: ";
    for (int d = 0; d < kv_print; ++d) {
        std::cout << k->get_as_float_flat(0 * k->shape[1] + d) << (d+1<kv_print?",":"");
    }
    std::cout << std::endl;
    std::cout << "[MHA] raw v[0,:" << kv_print << "]: ";
    for (int d = 0; d < kv_print; ++d) {
        std::cout << v->get_as_float_flat(0 * v->shape[1] + d) << (d+1<kv_print?",":"");
    }
    std::cout << std::endl;

    // Compute simple max-abs for v
    float vmax = 0.0f;
    for (int i = 0; i < v->shape[0]; ++i) {
        for (int j = 0; j < v->shape[1]; ++j) {
            float val = v->get_as_float_flat((size_t)i * v->shape[1] + j);
            vmax = std::max(vmax, std::fabs(val));
        }
    }
    std::cout << "[MHA] v max|val|=" << vmax << std::endl;
    
    // Reshape to [seq_len, num_heads, head_dim]
    auto q_reshaped = q->reshape_view({seq_len, num_heads, head_dim});
    auto k_reshaped = k->reshape_view({seq_len, num_kv_heads, head_dim});
    auto v_reshaped = v->reshape_view({seq_len, num_kv_heads, head_dim});
    std::cout << "[MHA] Matmul done: q=[" << q->shape[0] << "," << q->shape[1]
              << "] k=[" << k->shape[0] << "," << k->shape[1]
              << "] v=[" << v->shape[0] << "," << v->shape[1] << "]" << std::endl;
    
    // Apply RMSNorm to Q and K
    std::cout << "[MHA] Pre-norm shapes: q_r=[" << q_reshaped->shape[0] << "," << q_reshaped->shape[1] << "," << q_reshaped->shape[2]
              << "] k_r=[" << k_reshaped->shape[0] << "," << k_reshaped->shape[1] << "," << k_reshaped->shape[2] << "]" << std::endl;
    q_reshaped = q_norm->forward(q_reshaped);
    k_reshaped = k_norm->forward(k_reshaped);
    std::cout << "[MHA] Post-norm" << std::endl;
    
    // Apply rotary position embedding
    auto [cos, sin] = rotary_emb->forward(seq_len, ctx);
    std::cout << "[MHA] Rotary forward: cos=[" << cos->shape[0] << "," << cos->shape[1] << "]" << std::endl;
    try {
        q_reshaped = rotary_emb->apply_rotary_pos_emb(q_reshaped, cos, sin);
        std::cout << "[MHA] RoPE applied to Q" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[MHA] RoPE apply failed for Q: " << e.what() << std::endl;
        throw;
    }
    try {
        k_reshaped = rotary_emb->apply_rotary_pos_emb(k_reshaped, cos, sin);
        std::cout << "[MHA] RoPE applied to K" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[MHA] RoPE apply failed for K: " << e.what() << std::endl;
        throw;
    }
    std::cout << "[MHA] Rotary applied" << std::endl;
    
    // Debug: print first few q/k/v values after norm+RoPE for i=0,h=0
    std::cout << "[MHA] q[0,0,:]: ";
    for (int d=0; d<head_dim; ++d) {
        std::cout << q_reshaped->get_as_float_flat(0 * num_heads * head_dim + 0 * head_dim + d) << (d+1<head_dim?",":"");
    }
    std::cout << std::endl;
    std::cout << "[MHA] k[0,0,:]: ";
    for (int d=0; d<head_dim; ++d) {
        std::cout << k_reshaped->get_as_float_flat(0 * num_kv_heads * head_dim + 0 * head_dim + d) << (d+1<head_dim?",":"");
    }
    std::cout << std::endl;
    std::cout << "[MHA] v[0,0,:]: ";
    for (int d=0; d<head_dim; ++d) {
        std::cout << v_reshaped->get_as_float_flat(0 * num_kv_heads * head_dim + 0 * head_dim + d) << (d+1<head_dim?",":"");
    }
    std::cout << std::endl;
    
    // Compute attention scores
    // For simplicity, we'll use a basic attention implementation
    // In practice, you'd want to use more optimized kernels
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn_output = Tensor::create(ctx, {seq_len, num_heads, head_dim}, DType::FLOAT32);
    
    // Simplified attention computation (not optimized)
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h % num_kv_heads;  // Handle grouped query attention
        
        for (int i = 0; i < seq_len; ++i) {
            std::vector<float> scores(seq_len);
            
            // Compute attention scores
            for (int j = 0; j <= i; ++j) {  // Causal mask
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    float qv = q_reshaped->get_as_float_flat(i * num_heads * head_dim + h * head_dim + d);
                    float kvv = k_reshaped->get_as_float_flat(j * num_kv_heads * head_dim + kv_h * head_dim + d);
                    score += qv * kvv;
                }
                scores[j] = score * scale;
            }
            
            // Softmax over scores[0..i]
            float max_score = -1e30f;
            for (int j = 0; j <= i; ++j) max_score = std::max(max_score, scores[j]);
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (int j = 0; j <= i; ++j) scores[j] /= sum_exp;
            
            // Weighted sum of V
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    val += scores[j] * v_reshaped->get_as_float_flat(j * num_kv_heads * head_dim + kv_h * head_dim + d);
                }
                attn_output->set_from_float_flat(i * num_heads * head_dim + h * head_dim + d, val);
            }
        }
    }
    
    // Reshape back and project output
    auto attn_flat = attn_output->reshape_view({seq_len, hidden_size});
    auto out = Tensor::matmul_blocked_mt(attn_flat, o_proj);
    
    // Debug: print a slice of attn_flat and out
    int out_print = std::min(hidden_size, 8);
    std::cout << "[MHA] attn_flat[0,:" << out_print << "]: ";
    for (int d = 0; d < out_print; ++d) {
        std::cout << attn_flat->get_as_float_flat(0 * hidden_size + d) << (d+1<out_print?",":"");
    }
    std::cout << std::endl;
    std::cout << "[MHA] o_proj out[0,:" << out_print << "]: ";
    for (int d = 0; d < out_print; ++d) {
        std::cout << out->get_as_float_flat(0 * hidden_size + d) << (d+1<out_print?",":"");
    }
    std::cout << std::endl;

    return out;
}

// Implement Expert, SparseMoEBlock, and DecoderLayer forwards
TensorPtr Expert::forward(const TensorPtr& x) {
    auto ctx = x->ctx.lock();
    if (!ctx) throw std::runtime_error("Expert: ctx expired");
    // Simple MLP: up -> down (activation omitted for simplicity)
    auto up = Tensor::matmul_blocked_mt(x, up_proj);
    auto down = Tensor::matmul_blocked_mt(up, down_proj);
    return down;
}

TensorPtr SparseMoEBlock::forward(const TensorPtr& x) {
    auto ctx = x->ctx.lock();
    if (!ctx) throw std::runtime_error("SparseMoEBlock: ctx expired");
    // Router scores: [seq_len, num_experts]
    auto scores = Tensor::matmul_blocked_mt(x, gate);
    int seq_len = x->shape[0];
    int hidden_size = x->shape[1];
    auto output = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    
    // Top-1 routing per token for simplicity
    for (int i = 0; i < seq_len; ++i) {
        int best = 0;
        float best_score = -1e30f;
        for (int e = 0; e < num_experts; ++e) {
            float s = scores->get_as_float_flat((size_t)i * num_experts + e);
            if (s > best_score) { best_score = s; best = e; }
        }
        // x[i,:] view
        auto x_i = x->slice_view({i, 0}, {1, hidden_size});
        auto routed = experts[best]->forward(x_i);
        for (int h = 0; h < hidden_size; ++h) {
            output->set_from_float_flat((size_t)i * hidden_size + h, routed->get_as_float_flat(h));
        }
    }
    return output;
}

TensorPtr DecoderLayer::forward(const TensorPtr& hidden_states, int seq_len) {
    auto ctx = hidden_states->ctx.lock();
    if (!ctx) throw std::runtime_error("DecoderLayer: ctx expired");
    
    // Pre-attention norm
    auto h1 = input_layernorm->forward(hidden_states);
    // Self attention
    auto attn = self_attn->forward(h1, seq_len);
    
    // Residual add
    auto h_attn = Tensor::create(ctx, hidden_states->shape, DType::FLOAT32);
    int rows = hidden_states->shape[0];
    int cols = hidden_states->shape[1];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float a = hidden_states->get_as_float_flat((size_t)i * cols + j);
            float b = attn->get_as_float_flat((size_t)i * cols + j);
            h_attn->set_from_float_flat((size_t)i * cols + j, a + b);
        }
    }
    
    // Post-attention norm
    auto h2 = post_attention_layernorm->forward(h_attn);
    // MoE MLP
    auto mlp_out = mlp->forward(h2);
    
    // Residual add
    auto out = Tensor::create(ctx, hidden_states->shape, DType::FLOAT32);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float a = h_attn->get_as_float_flat((size_t)i * cols + j);
            float b = mlp_out->get_as_float_flat((size_t)i * cols + j);
            out->set_from_float_flat((size_t)i * cols + j, a + b);
        }
    }
    return out;
}

Qwen3VLTextModel::Qwen3VLTextModel(const TensorPtr& embed_tokens,
                                  const std::vector<std::shared_ptr<DecoderLayer>>& layers,
                                  std::shared_ptr<RMSNorm> norm,
                                  int vocab_size, int hidden_size, int num_layers, 
                                  int num_heads, int num_kv_heads)
    : embed_tokens(embed_tokens), layers(layers), norm(norm),
      vocab_size(vocab_size), hidden_size(hidden_size), num_layers(num_layers),
      num_heads(num_heads), num_kv_heads(num_kv_heads) {
    
    rotary_emb = std::make_shared<RotaryEmbedding>(hidden_size / num_heads);
}

TensorPtr Qwen3VLTextModel::forward(const std::vector<int>& input_ids) {
    auto ctx = embed_tokens->ctx.lock();
    if (!ctx) throw std::runtime_error("Qwen3VLTextModel: ctx expired");
    std::cout << "[Forward] Start: input_ids=" << input_ids.size() << std::endl;
    
    int seq_len = input_ids.size();
    // Mark arena to release temporaries after forward
    size_t arena_mark = ctx->mark();
    
    // Embedding lookup
    auto hidden_states = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    std::cout << "[Forward] Alloc hidden_states: shape=[" << seq_len << "," << hidden_size << "]" << std::endl;
    
    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_ids[i];
        for (int h = 0; h < hidden_size; ++h) {
            float val = embed_tokens->get_as_float_flat(token_id * hidden_size + h);
            hidden_states->set_from_float_flat(i * hidden_size + h, val);
        }
    }
    std::cout << "[Forward] Embedding done: seq_len=" << seq_len
              << " hidden_size=" << hidden_size << std::endl;
    
    // Pass through decoder layers
    for (size_t li = 0; li < layers.size(); ++li) {
        std::cout << "[Forward] Layer " << li << " start" << std::endl;
        hidden_states = layers[li]->forward(hidden_states, seq_len);
        std::cout << "[Forward] Layer " << li << " done" << std::endl;
    }
    
    // Final layer norm
    hidden_states = norm->forward(hidden_states);
    std::cout << "[Forward] Final norm done" << std::endl;

    // Copy result to a fresh tensor and release temporaries to avoid arena growth
    auto result = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    for (size_t i = 0; i < hidden_states->nelements(); ++i) {
        result->set_from_float_flat(i, hidden_states->get_as_float_flat(i));
    }
    ctx->release_to(arena_mark);
    return result;
}

// WeightLoader implementation
std::shared_ptr<Qwen3VLTextModel> WeightLoader::build_model() {
    // Model configuration based on Qwen3VL config.json
    const int vocab_size = 151936;
    const int hidden_size = 2048;  // Corrected from config.json
    const int num_layers = 48;     // Corrected from config.json
    const int num_heads = 32;
    const int num_kv_heads = 4;
    const int intermediate_size = 6144;  // Corrected from config.json
    const int num_experts = 128;
    const int top_k = 8;
    
    // Get embedding weights
    auto embed_tokens = get_weight("model.embed_tokens.weight");
    if (!embed_tokens) {
        throw std::runtime_error("WeightLoader: embed_tokens.weight not found");
    }
    
    // Build decoder layers
    std::vector<std::shared_ptr<DecoderLayer>> layers;
    
    for (int i = 0; i < num_layers; ++i) {
        std::string layer_prefix = "model.layers." + std::to_string(i) + ".";
        
        // Attention weights
        auto q_proj_raw = get_weight(layer_prefix + "self_attn.q_proj.weight");
        auto k_proj_raw = get_weight(layer_prefix + "self_attn.k_proj.weight");
        auto v_proj_raw = get_weight(layer_prefix + "self_attn.v_proj.weight");
        auto o_proj_raw = get_weight(layer_prefix + "self_attn.o_proj.weight");
        auto q_norm_weight = get_weight(layer_prefix + "self_attn.q_norm.weight");
        auto k_norm_weight = get_weight(layer_prefix + "self_attn.k_norm.weight");
        
        if (!q_proj_raw || !k_proj_raw || !v_proj_raw || !o_proj_raw || !q_norm_weight || !k_norm_weight) {
            std::cerr << "Warning: Missing attention weights for layer " << i << std::endl;
            continue;
        }
        // Fix orientation to [hidden_size, out_dim]
        auto q_proj = ensure_linear_weight_or_transpose(q_proj_raw, hidden_size, hidden_size);
        auto k_proj = ensure_linear_weight_or_transpose(k_proj_raw, hidden_size);
        auto v_proj = ensure_linear_weight_or_transpose(v_proj_raw, hidden_size);
        auto o_proj = ensure_linear_weight_or_transpose(o_proj_raw, hidden_size, hidden_size);
        
        // Create attention layer
        auto attention = std::make_shared<MultiHeadAttention>(
            q_proj, k_proj, v_proj, o_proj,
            q_norm_weight, k_norm_weight,
            num_heads, num_kv_heads, hidden_size
        );
        
        // MoE weights - this model uses merged expert weights
        auto gate_weight_raw = get_weight(layer_prefix + "mlp.gate.weight");
        auto experts_gate_up_proj = get_weight(layer_prefix + "mlp.experts.gate_up_proj");
        auto experts_down_proj = get_weight(layer_prefix + "mlp.experts.down_proj");
        
        if (!gate_weight_raw || !experts_gate_up_proj || !experts_down_proj) {
            std::cerr << "Warning: Missing MoE weights for layer " << i << std::endl;
            continue;
        }
        
        auto gate_weight = ensure_linear_weight_or_transpose(gate_weight_raw, hidden_size);
        
        // For merged expert weights, we create a simplified expert structure
        // The actual expert selection and computation will be handled differently
        std::vector<std::shared_ptr<Expert>> experts;
        
        // Create a single "merged expert" that represents all experts
        // This is a simplified approach - in a full implementation, you'd need
        // to properly handle the merged weight tensors
        auto merged_expert = std::make_shared<Expert>(
            experts_gate_up_proj,  // Combined gate and up projections
            experts_gate_up_proj,  // Reuse for up_proj (will need proper slicing)
            experts_down_proj      // Down projection
        );
        experts.push_back(merged_expert);
        
        // Build MoE block
        auto moe = std::make_shared<SparseMoEBlock>(gate_weight, experts, top_k);
        
        // Layer norms
        auto input_layernorm_weight = get_weight(layer_prefix + "input_layernorm.weight");
        auto post_attention_layernorm_weight = get_weight(layer_prefix + "post_attention_layernorm.weight");
        
        if (!input_layernorm_weight || !post_attention_layernorm_weight) {
            std::cerr << "Warning: Missing layer norm weights for layer " << i << std::endl;
            continue;
        }
        
        auto input_layernorm = std::make_shared<RMSNorm>(input_layernorm_weight);
        auto post_attention_layernorm = std::make_shared<RMSNorm>(post_attention_layernorm_weight);
        
        // Create decoder layer
        auto decoder_layer = std::make_shared<DecoderLayer>(
            attention, moe, input_layernorm, post_attention_layernorm
        );
        
        layers.push_back(decoder_layer);
    }
    
    // Final layer norm
    auto norm_weight = get_weight("model.norm.weight");
    if (!norm_weight) {
        throw std::runtime_error("WeightLoader: model.norm.weight not found");
    }
    auto norm = std::make_shared<RMSNorm>(norm_weight);
    
    // Create model
    auto model = std::make_shared<Qwen3VLTextModel>(
        embed_tokens, layers, norm,
        vocab_size, hidden_size, num_layers, num_heads, num_kv_heads
    );
    
    std::cout << "Built Qwen3VL model with " << layers.size() << " layers" << std::endl;
    
    return model;
}

} // namespace ow::nn