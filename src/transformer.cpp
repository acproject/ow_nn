#include "../include/transformer.h"
#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ow::nn {

// Verbosity gates forward declarations
static bool ow_verbose_rope();
static bool ow_verbose_moe();

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
        // Only print warning once per unique dimension combination to reduce log spam
        static std::set<std::string> warned_combinations;
        std::string key = std::to_string(r0) + "x" + std::to_string(r1) + "->" + 
                         std::to_string(expected_in_dim) + "x" + std::to_string(expected_out_dim<0? r0: expected_out_dim);
        if (warned_combinations.find(key) == warned_combinations.end()) {
            std::cerr << "[Weight] Transposing weight: expected ["
                      << expected_in_dim << "," << (expected_out_dim<0? r0: expected_out_dim)
                      << "] got [" << r0 << "," << r1 << "] (further similar warnings suppressed)" << std::endl;
            warned_combinations.insert(key);
        }
        // Actually perform the transpose for matvec compatibility
        auto ctx = W->ctx.lock();
        if (!ctx) throw std::runtime_error("ensure_linear_weight_or_transpose: ctx expired");
        auto W_T = Tensor::create(ctx, {r1, r0}, W->dtype);
        for (int i = 0; i < r0; ++i) {
            for (int j = 0; j < r1; ++j) {
                W_T->set_from_float_flat(j * r0 + i, W->get_as_float_flat(i * r1 + j));
            }
        }
        return W_T;
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
            if (!std::isfinite(val)) val = 0.0f;
            sum_sq += val * val;
        }
        float denom = sum_sq / hidden_size + eps;
        if (!std::isfinite(denom) || denom <= 0.0f) denom = std::max(eps, 1e-6f);
        float rms = std::sqrt(denom);
        if (!std::isfinite(rms) || rms < 1e-6f) rms = 1e-6f;
        
        // Normalize and scale
        for (int j = 0; j < hidden_size; ++j) {
            float val = x->get_as_float_flat(i * hidden_size + j);
            if (!std::isfinite(val)) val = 0.0f;
            float weight_val = weight->get_as_float_flat(j);
            if (!std::isfinite(weight_val)) weight_val = 1.0f;
            float normalized = (val / rms) * weight_val;
            if (!std::isfinite(normalized)) normalized = 0.0f;
            output->set_from_float_flat(i * hidden_size + j, normalized);
        }
    }
    
    return output;
}

// RotaryEmbedding implementation (declarations are in transformer.h)
// Add missing RotaryEmbedding constructor and forward
RotaryEmbedding::RotaryEmbedding(int dim, int max_seq_len, float base, 
                               float attention_scaling, const std::vector<int>& mrope_section) 
    : dim(dim), max_seq_len(max_seq_len), base(base), 
      attention_scaling(attention_scaling), mrope_section(mrope_section), cos_cached(nullptr), sin_cached(nullptr) {}

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
                if (ow_verbose_rope() && i == 0 && h == 0 && d == 0) {
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

// MRoPE-Interleave implementation for 3D position encoding
std::pair<TensorPtr, TensorPtr> RotaryEmbedding::forward_mrope(const TensorPtr& position_ids, const std::shared_ptr<Context>& ctx) {
    if (!ctx) throw std::runtime_error("RotaryEmbedding::forward_mrope: ctx expired");
    
    // position_ids shape: [3, batch_size, seq_len] for (T, H, W)
    if (position_ids->shape.size() != 3 || position_ids->shape[0] != 3) {
        throw std::runtime_error("MRoPE position_ids must have shape [3, batch_size, seq_len]");
    }
    
    int batch_size = position_ids->shape[1];
    int seq_len = position_ids->shape[2];
    
    // Create frequency tensor for each dimension (T, H, W)
    auto freqs = Tensor::create(ctx, {3, seq_len, dim / 2}, DType::FLOAT32);
    
    // Calculate frequencies for each dimension
    for (int dim_idx = 0; dim_idx < 3; ++dim_idx) {
        for (int pos = 0; pos < seq_len; ++pos) {
            // Get position for this dimension
            int position = static_cast<int>(position_ids->get_as_float_flat(dim_idx * seq_len + pos));
            
            for (int i = 0; i < dim / 2; ++i) {
                float inv_freq = 1.0f / std::pow(base, (2.0f * i) / dim);
                float freq = position * inv_freq * attention_scaling;
                freqs->set_from_float_flat(dim_idx * seq_len * (dim / 2) + pos * (dim / 2) + i, freq);
            }
        }
    }
    
    // Apply interleaved MRoPE to reorganize frequencies
    auto interleaved_freqs = apply_interleaved_mrope(freqs, mrope_section);
    
    // Calculate cos and sin from interleaved frequencies
    auto cos_vals = Tensor::create(ctx, {seq_len, dim / 2}, DType::FLOAT32);
    auto sin_vals = Tensor::create(ctx, {seq_len, dim / 2}, DType::FLOAT32);
    
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < dim / 2; ++i) {
            float freq = interleaved_freqs->get_as_float_flat(pos * (dim / 2) + i);
            cos_vals->set_from_float_flat(pos * (dim / 2) + i, std::cos(freq));
            sin_vals->set_from_float_flat(pos * (dim / 2) + i, std::sin(freq));
        }
    }
    
    return {cos_vals, sin_vals};
}

TensorPtr RotaryEmbedding::apply_interleaved_mrope(const TensorPtr& freqs, const std::vector<int>& mrope_section) {
    auto ctx = freqs->ctx.lock();
    if (!ctx) throw std::runtime_error("apply_interleaved_mrope: ctx expired");
    
    // freqs shape: [3, seq_len, dim/2]
    int seq_len = freqs->shape[1];
    int half_dim = freqs->shape[2];
    
    // Validate mrope_section
    if (mrope_section.size() != 3) {
        throw std::runtime_error("mrope_section must have 3 elements [t_section, h_section, w_section]");
    }
    
    int t_section = mrope_section[0];
    int h_section = mrope_section[1]; 
    int w_section = mrope_section[2];
    
    if (t_section + h_section + w_section != half_dim) {
        throw std::runtime_error("Sum of mrope_section must equal dim/2");
    }
    
    // Create output tensor with interleaved layout
    auto output = Tensor::create(ctx, {seq_len, half_dim}, DType::FLOAT32);
    
    // Apply interleaved pattern: [THTHWHTHW...TT]
    for (int pos = 0; pos < seq_len; ++pos) {
        int out_idx = 0;
        
        // Interleave T, H, W frequencies
        int min_section = std::min({t_section, h_section, w_section});
        for (int i = 0; i < min_section; ++i) {
            // T frequency
            float t_freq = freqs->get_as_float_flat(0 * seq_len * half_dim + pos * half_dim + i);
            output->set_from_float_flat(pos * half_dim + out_idx++, t_freq);
            
            // H frequency  
            float h_freq = freqs->get_as_float_flat(1 * seq_len * half_dim + pos * half_dim + i);
            output->set_from_float_flat(pos * half_dim + out_idx++, h_freq);
            
            // W frequency
            float w_freq = freqs->get_as_float_flat(2 * seq_len * half_dim + pos * half_dim + i);
            output->set_from_float_flat(pos * half_dim + out_idx++, w_freq);
        }
        
        // Add remaining T frequencies if t_section > min_section
        for (int i = min_section; i < t_section; ++i) {
            float t_freq = freqs->get_as_float_flat(0 * seq_len * half_dim + pos * half_dim + i);
            output->set_from_float_flat(pos * half_dim + out_idx++, t_freq);
        }
        
        // Add remaining H frequencies if h_section > min_section  
        for (int i = min_section; i < h_section; ++i) {
            float h_freq = freqs->get_as_float_flat(1 * seq_len * half_dim + pos * half_dim + i);
            output->set_from_float_flat(pos * half_dim + out_idx++, h_freq);
        }
        
        // Add remaining W frequencies if w_section > min_section
        for (int i = min_section; i < w_section; ++i) {
            float w_freq = freqs->get_as_float_flat(2 * seq_len * half_dim + pos * half_dim + i);
            output->set_from_float_flat(pos * half_dim + out_idx++, w_freq);
        }
    }
    
    return output;
}

// MultiHeadAttention implementation
// Add environment-gated verbosity for MHA/KV debug
static bool ow_verbose_mha() {
    const char* v = std::getenv("OWNN_VERBOSE_MHA");
    return v != nullptr && v[0] != '0';
}
// Add separate gates for RoPE and MoE
static bool ow_verbose_rope() {
    const char* v = std::getenv("OWNN_VERBOSE_ROPE");
    return v != nullptr && v[0] != '0';
}
static bool ow_verbose_moe() {
    const char* v = std::getenv("OWNN_VERBOSE_MOE");
    return v != nullptr && v[0] != '0';
}

static float ow_env_rope_base() {
    const char* v = std::getenv("OWNN_ROPE_BASE");
    if (!v || v[0] == '\0') v = std::getenv("ROPE_THETA");
    if (!v || v[0] == '\0') return 10000.0f;
    char* end = nullptr;
    float val = std::strtof(v, &end);
    if (end == v || !std::isfinite(val) || val <= 0.0f) return 10000.0f;
    return val;
}

MultiHeadAttention::MultiHeadAttention(const TensorPtr& q_proj, const TensorPtr& k_proj, 
                                     const TensorPtr& v_proj, const TensorPtr& o_proj,
                                     const TensorPtr& q_norm_weight, const TensorPtr& k_norm_weight,
                                     int num_heads, int num_kv_heads, int hidden_size)
    : q_proj(q_proj), k_proj(k_proj), v_proj(v_proj), o_proj(o_proj),
      num_heads(num_heads), num_kv_heads(num_kv_heads), hidden_size(hidden_size) {
    
    // Infer head_dim from weight shapes to avoid reshape mismatch (supports transposed weights)
    auto infer_out_dim = [&](const TensorPtr& W) -> int {
        if (!W || W->shape.size() != 2) return hidden_size;
        // If W is [in_dim, out_dim]
        if (W->shape[0] == hidden_size) return W->shape[1];
        // If W is [out_dim, in_dim]
        if (W->shape[1] == hidden_size) return W->shape[0];
        // Fallback
        return W->shape[1];
    };
    int q_out = infer_out_dim(q_proj);
    int k_out = infer_out_dim(k_proj);
    int v_out = infer_out_dim(v_proj);
    int hd_q = (num_heads > 0 && q_out % num_heads == 0) ? (q_out / num_heads) : -1;
    int hd_k = (num_kv_heads > 0 && k_out % num_kv_heads == 0) ? (k_out / num_kv_heads) : -1;
    int hd_v = (num_kv_heads > 0 && v_out % num_kv_heads == 0) ? (v_out / num_kv_heads) : -1;
    if (hd_q > 0 && hd_k > 0 && hd_q != hd_k && ow_verbose_mha()) {
        std::cout << "[MHA] head_dim mismatch q=" << hd_q << " k=" << hd_k
                  << ", prefer q-based head_dim" << std::endl;
    }
    int inferred_hd = (hd_q > 0) ? hd_q : (hd_k > 0 ? hd_k : (hidden_size / std::max(1, num_heads)));
    head_dim = inferred_hd;

    // Auto-correct num_kv_heads based on actual K/V projection sizes
    int kv_from_k = (head_dim > 0 && k_out % head_dim == 0) ? (k_out / head_dim) : -1;
    int kv_from_v = (head_dim > 0 && v_out % head_dim == 0) ? (v_out / head_dim) : -1;
    int kv_target = kv_from_k > 0 ? kv_from_k : kv_from_v;
    if (kv_target > 0 && kv_target != num_kv_heads) {
        if (ow_verbose_mha()) {
            std::cout << "[MHA] Adjust num_kv_heads from " << num_kv_heads
                      << " to " << kv_target << " based on k/v dims (k_out=" << k_out
                      << ", v_out=" << v_out << ", head_dim=" << head_dim << ")" << std::endl;
        }
        this->num_kv_heads = kv_target;
    }

    if (ow_verbose_mha()) {
        std::cout << "[MHA] head_dim inference: hidden_size=" << hidden_size
                  << " q_out=" << q_out << " k_out=" << k_out << " v_out=" << v_out
                  << " -> head_dim=" << head_dim << " (num_heads=" << num_heads
                  << ", num_kv_heads=" << this->num_kv_heads << ")" << std::endl;
    }
    q_norm = std::make_shared<RMSNorm>(q_norm_weight);
    k_norm = std::make_shared<RMSNorm>(k_norm_weight);
    {
        float rope_base = ow_env_rope_base();
        if (ow_verbose_rope()) {
            std::cout << "[RoPE] base=" << rope_base << " dim=" << head_dim << std::endl;
        }
        rotary_emb = std::make_shared<RotaryEmbedding>(head_dim, 2048, rope_base);
    }

    // KV cache state
    cache_len = 0;
    max_seq_len = 0;
}

// Initialize KV cache (persistent across forwards)
void MultiHeadAttention::init_cache(const std::shared_ptr<Context>& ctx, int max_len) {
    if (!ctx) throw std::runtime_error("MultiHeadAttention::init_cache: ctx expired");
    if (max_len <= 0) max_len = 2048;
    max_seq_len = max_len;
    k_cache = Tensor::create(ctx, {max_seq_len, num_kv_heads, head_dim}, DType::FLOAT32);
    v_cache = Tensor::create(ctx, {max_seq_len, num_kv_heads, head_dim}, DType::FLOAT32);
    cache_len = 0;
}

TensorPtr MultiHeadAttention::forward(const TensorPtr& hidden_states, int seq_len) {
    auto ctx = hidden_states->ctx.lock();
    if (!ctx) throw std::runtime_error("MultiHeadAttention: ctx expired");
    if (ow_verbose_mha()) {
        std::cout << "[MHA] Start: seq_len=" << seq_len
                  << " hidden_size=" << hidden_size
                  << " q_proj=[" << q_proj->shape[0] << "," << q_proj->shape[1] << "]"
                  << " k_proj=[" << k_proj->shape[0] << "," << k_proj->shape[1] << "]"
                  << " v_proj=[" << v_proj->shape[0] << "," << v_proj->shape[1] << "]"
                  << " v_proj.dtype=" << (int)v_proj->dtype
                  << std::endl;
    }

    // Ensure KV cache is initialized and persistent
    if (!k_cache || !v_cache) {
        init_cache(ctx, std::max(seq_len, rotary_emb->max_seq_len));
        if (ow_verbose_mha()) {
            std::cout << "[KV] Initialized cache: max_seq_len=" << max_seq_len << std::endl;
        }
    }
    // Reset cache when sequence length shrinks (new prompt)
    if (seq_len < cache_len) {
        if (ow_verbose_mha()) {
            std::cout << "[KV] Reset cache_len from " << cache_len << " to 0 (new prompt)" << std::endl;
        }
        cache_len = 0;
    }

    // Print a slice of hidden_states row 0
    int hs_print = std::min(hidden_size, 8);
    if (ow_verbose_mha()) {
        std::cout << "[MHA] hidden_states[0,:" << hs_print << "]: ";
        for (int h = 0; h < hs_print; ++h) {
            std::cout << hidden_states->get_as_float_flat(0 * hidden_size + h) << (h+1<hs_print?",":"");
        }
        std::cout << std::endl;
    }

    // hidden_states: [seq_len, hidden_size]
    // Project Q for all tokens (kept simple); K/V update incrementally using cache
    auto q = Tensor::matmul_cache_friendly(hidden_states, q_proj);

    // Sanitize possible NaN/Inf produced by conversions or matmul
    auto sanitize_tensor = [](const TensorPtr& t, const std::string& name = "") {
        if (!t) return;
        size_t n = t->nelements();
        size_t nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < n; ++i) {
            float v = t->get_as_float_flat(i);
            if (std::isnan(v)) {
                nan_count++;
                t->set_from_float_flat(i, 0.0f);
            } else if (std::isinf(v)) {
                inf_count++;
                t->set_from_float_flat(i, std::copysign(1e6f, v));
            }
        }
        if (nan_count > 0 || inf_count > 0) {
            std::cerr << "[SANITIZE] " << name << ": fixed " << nan_count << " NaN(s) and " 
                      << inf_count << " Inf(s) out of " << n << " elements" << std::endl;
        }
    };
    sanitize_tensor(q, "q_proj");

    // Reshape Q and apply RMSNorm + RoPE
    auto q_reshaped = q->reshape_view({seq_len, num_heads, head_dim});
    q_reshaped = q_norm->forward(q_reshaped);
    auto [cos, sin] = rotary_emb->forward(seq_len, ctx);
    q_reshaped = rotary_emb->apply_rotary_pos_emb(q_reshaped, cos, sin);

    // Incrementally compute K/V for new tokens and update caches (normalized + RoPE K, raw V)
    int start = cache_len;
    for (int i = start; i < seq_len; ++i) {
        // slice current token
        auto x_i = hidden_states->slice_view({i, 0}, {1, hidden_size});
        
        // Debug: print x_i shape
        std::cout << "[MHA DBG] x_i shape: [";
        for (size_t j = 0; j < x_i->shape.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << x_i->shape[j];
        }
        std::cout << "], k_proj shape: [";
        for (size_t j = 0; j < k_proj->shape.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << k_proj->shape[j];
        }
        std::cout << "]" << std::endl;
        
        // matvec for K and V (panel packing + blocking)
        auto k_i = Tensor::matvec_blocked_mt(x_i, k_proj);
        auto v_i = Tensor::matvec_blocked_mt(x_i, v_proj);
        sanitize_tensor(k_i, "k_proj[i]");
        sanitize_tensor(v_i, "v_proj[i]");
        // Debug current KV dims before reshape
        if (ow_verbose_mha()) {
            std::cout << "[MHA DBG] num_kv_heads=" << num_kv_heads
                      << " head_dim=" << head_dim
                      << " k_i out_dim=" << (k_i->shape.size()>=2?k_i->shape[1]:-1)
                      << " v_i out_dim=" << (v_i->shape.size()>=2?v_i->shape[1]:-1)
                      << std::endl;
        }
        // reshape to [1, num_kv_heads, head_dim]
        auto k_i_r = k_i->reshape_view({1, num_kv_heads, head_dim});
        auto v_i_r = v_i->reshape_view({1, num_kv_heads, head_dim});
        // k-norm
        k_i_r = k_norm->forward(k_i_r);
        // Apply RoPE manually for position i and store to cache
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; d += 2) {
                size_t src1 = 0 * num_kv_heads * head_dim + h * head_dim + d;
                float x1 = k_i_r->get_as_float_flat(src1);
                float x2 = k_i_r->get_as_float_flat(src1 + 1);
                size_t cos_idx = (size_t)i * (head_dim / 2) + (size_t)(d / 2);
                float cos_val = cos->get_as_float_flat(cos_idx);
                float sin_val = sin->get_as_float_flat(cos_idx);
                float new_x1 = x1 * cos_val - x2 * sin_val;
                float new_x2 = x1 * sin_val + x2 * cos_val;
                size_t dst1 = (size_t)i * num_kv_heads * head_dim + (size_t)h * head_dim + d;
                k_cache->set_from_float_flat(dst1, new_x1);
                k_cache->set_from_float_flat(dst1 + 1, new_x2);
            }
        }
        // Store V to cache
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float vv = v_i_r->get_as_float_flat(0 * num_kv_heads * head_dim + h * head_dim + d);
                size_t dst = (size_t)i * num_kv_heads * head_dim + (size_t)h * head_dim + d;
                v_cache->set_from_float_flat(dst, vv);
            }
        }
    }
    cache_len = seq_len;

    // Debug: print first few K/V cache values at position 0
    int kv_print = std::min(head_dim, 8);
    if (ow_verbose_mha()) {
        std::cout << "[KV] k_cache[0,0,:]: ";
        for (int d = 0; d < kv_print; ++d) {
            std::cout << k_cache->get_as_float_flat(0 * num_kv_heads * head_dim + 0 * head_dim + d) << (d+1<kv_print?",":"");
        }
        std::cout << std::endl;
        std::cout << "[KV] v_cache[0,0,:]: ";
        for (int d = 0; d < kv_print; ++d) {
            std::cout << v_cache->get_as_float_flat(0 * num_kv_heads * head_dim + 0 * head_dim + d) << (d+1<kv_print?",":"");
        }
        std::cout << std::endl;
    }
    
    // Compute attention scores with improved numerical stability using cached K/V
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn_output = Tensor::create(ctx, {seq_len, num_heads, head_dim}, DType::FLOAT32);
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h % num_kv_heads;  // grouped query attention
        for (int i = 0; i < seq_len; ++i) {
            std::vector<float> scores(seq_len);
            for (int j = 0; j <= i; ++j) {  // causal mask
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    float qv = q_reshaped->get_as_float_flat(i * num_heads * head_dim + h * head_dim + d);
                    float kvv = k_cache->get_as_float_flat(j * num_kv_heads * head_dim + kv_h * head_dim + d);
                    if (!std::isfinite(qv)) qv = 0.0f;
                    if (!std::isfinite(kvv)) kvv = 0.0f;
                    score += qv * kvv;
                }
                score *= scale;
                if (!std::isfinite(score)) score = -50.0f;
                if (score > 50.0f) score = 50.0f;
                if (score < -50.0f) score = -50.0f;
                scores[j] = score;
            }
            float max_score = -std::numeric_limits<float>::infinity();
            for (int j = 0; j <= i; ++j) {
                if (std::isfinite(scores[j])) {
                    max_score = std::max(max_score, scores[j]);
                }
            }
            if (!std::isfinite(max_score)) max_score = 0.0f;
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float exp_val = std::exp(scores[j] - max_score);
                if (!std::isfinite(exp_val)) exp_val = 0.0f;
                scores[j] = exp_val;
                sum_exp += exp_val;
            }
            if (sum_exp <= 1e-10f) sum_exp = 1.0f;
            for (int j = 0; j <= i; ++j) scores[j] /= sum_exp;
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    val += scores[j] * v_cache->get_as_float_flat(j * num_kv_heads * head_dim + kv_h * head_dim + d);
                }
                attn_output->set_from_float_flat(i * num_heads * head_dim + h * head_dim + d, val);
            }
        }
    }
    
    // Reshape back and project output
    auto attn_flat = attn_output->reshape_view({seq_len, num_heads * head_dim});
    auto out = Tensor::matmul_cache_friendly(attn_flat, o_proj);
    sanitize_tensor(out, "o_proj");
     
    // Debug: print a slice of attn_flat and out
    int out_print = std::min(num_heads * head_dim, 8);
    if (ow_verbose_mha()) {
        std::cout << "[MHA] attn_flat[0,:" << out_print << "]: ";
        for (int d = 0; d < out_print; ++d) {
            std::cout << attn_flat->get_as_float_flat(0 * (num_heads * head_dim) + d) << (d+1<out_print?",":"");
        }
        std::cout << std::endl;
        std::cout << "[MHA] o_proj out[0,:" << out_print << "]: ";
        for (int d = 0; d < out_print; ++d) {
            std::cout << out->get_as_float_flat(0 * hidden_size + d) << (d+1<out_print?",":"");
        }
        std::cout << std::endl;
    }

    return out;
}

// MultiHeadAttention forward with MRoPE-Interleave support
TensorPtr MultiHeadAttention::forward(const TensorPtr& hidden_states, int seq_len, const TensorPtr& position_ids) {
    auto ctx = hidden_states->ctx.lock();
    if (!ctx) throw std::runtime_error("MultiHeadAttention: ctx expired");
    
    // Sanitize possible NaN/Inf produced by conversions or matmul
    auto sanitize_tensor = [](const TensorPtr& t, const std::string& name = "") {
        if (!t) return;
        size_t n = t->nelements();
        size_t nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < n; ++i) {
            float v = t->get_as_float_flat(i);
            if (std::isnan(v)) {
                nan_count++;
                t->set_from_float_flat(i, 0.0f);
            } else if (std::isinf(v)) {
                inf_count++;
                t->set_from_float_flat(i, std::copysign(1e6f, v));
            }
        }
        if (nan_count > 0 || inf_count > 0) {
            std::cerr << "[SANITIZE] " << name << ": fixed " << nan_count << " NaN(s) and " 
                      << inf_count << " Inf(s) out of " << n << " elements" << std::endl;
        }
    };
    
    // Use MRoPE-Interleave for position encoding
    auto [cos, sin] = rotary_emb->forward_mrope(position_ids, ctx);
    
    if (ow_verbose_mha()) {
        std::cout << "[MHA] Using MRoPE-Interleave: seq_len=" << seq_len
                  << " position_ids shape=[" << position_ids->shape[0] << "," 
                  << position_ids->shape[1] << "," << position_ids->shape[2] << "]" << std::endl;
    }

    // Ensure KV cache is initialized and persistent
    if (!k_cache || !v_cache) {
        init_cache(ctx, std::max(seq_len, rotary_emb->max_seq_len));
        if (ow_verbose_mha()) {
            std::cout << "[KV] Initialized cache: max_seq_len=" << max_seq_len << std::endl;
        }
    }

    // Incrementally compute K/V for new tokens and update caches (normalized + MRoPE K, raw V)
    int start = cache_len;
    for (int i = start; i < seq_len; ++i) {
        // slice current token
        auto x_i = hidden_states->slice_view({i, 0}, {1, hidden_size});
        // matvec for K and V (panel packing + blocking)
        auto k_i = Tensor::matvec_blocked_mt(x_i, k_proj);
        auto v_i = Tensor::matvec_blocked_mt(x_i, v_proj);
        sanitize_tensor(k_i, "k_proj[i]");
        sanitize_tensor(v_i, "v_proj[i]");
        
        // reshape to [1, num_kv_heads, head_dim]
        auto k_i_r = k_i->reshape_view({1, num_kv_heads, head_dim});
        auto v_i_r = v_i->reshape_view({1, num_kv_heads, head_dim});
        // k-norm
        k_i_r = k_norm->forward(k_i_r);
        
        // Apply MRoPE manually for position i and store to cache
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; d += 2) {
                size_t src1 = 0 * num_kv_heads * head_dim + h * head_dim + d;
                float x1 = k_i_r->get_as_float_flat(src1);
                float x2 = k_i_r->get_as_float_flat(src1 + 1);
                size_t cos_idx = (size_t)i * (head_dim / 2) + (size_t)(d / 2);
                float cos_val = cos->get_as_float_flat(cos_idx);
                float sin_val = sin->get_as_float_flat(cos_idx);
                float new_x1 = x1 * cos_val - x2 * sin_val;
                float new_x2 = x1 * sin_val + x2 * cos_val;
                size_t dst1 = (size_t)i * num_kv_heads * head_dim + (size_t)h * head_dim + d;
                k_cache->set_from_float_flat(dst1, new_x1);
                k_cache->set_from_float_flat(dst1 + 1, new_x2);
            }
        }
        // Store V to cache
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float vv = v_i_r->get_as_float_flat(0 * num_kv_heads * head_dim + h * head_dim + d);
                size_t dst = (size_t)i * num_kv_heads * head_dim + (size_t)h * head_dim + d;
                v_cache->set_from_float_flat(dst, vv);
            }
        }
    }
    cache_len = seq_len;

    // Compute Q for all tokens and apply MRoPE
    auto q_all = Tensor::matmul_cache_friendly(hidden_states, q_proj);
    sanitize_tensor(q_all, "q_proj");
    auto q_reshaped = q_all->reshape_view({seq_len, num_heads, head_dim});
    q_reshaped = q_norm->forward(q_reshaped);
    auto q_rope = rotary_emb->apply_rotary_pos_emb(q_reshaped, cos, sin);

    // Attention computation (same as standard forward)
    auto attn_flat = Tensor::create(ctx, {seq_len, num_heads * head_dim}, DType::FLOAT32);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h % num_kv_heads;
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    float q_val = q_rope->get_as_float_flat(i * num_heads * head_dim + h * head_dim + d);
                    float k_val = k_cache->get_as_float_flat(j * num_kv_heads * head_dim + kv_h * head_dim + d);
                    score += q_val * k_val;
                }
                score /= std::sqrt(static_cast<float>(head_dim));
                float attn_weight = std::exp(score);
                sum += attn_weight;
                
                for (int d = 0; d < head_dim; ++d) {
                    float v_val = v_cache->get_as_float_flat(j * num_kv_heads * head_dim + kv_h * head_dim + d);
                    size_t flat_idx = i * (num_heads * head_dim) + h * head_dim + d;
                    float current = attn_flat->get_as_float_flat(flat_idx);
                    attn_flat->set_from_float_flat(flat_idx, current + attn_weight * v_val);
                }
            }
            
            // Normalize
            for (int d = 0; d < head_dim; ++d) {
                size_t flat_idx = i * (num_heads * head_dim) + h * head_dim + d;
                float val = attn_flat->get_as_float_flat(flat_idx);
                attn_flat->set_from_float_flat(flat_idx, val / sum);
            }
        }
    }

    // Output projection
    auto out = Tensor::matmul_cache_friendly(attn_flat, o_proj);
    sanitize_tensor(out, "o_proj");

    return out;
}

// Implement Expert, SparseMoEBlock, and DecoderLayer forwards
TensorPtr Expert::forward(const TensorPtr& x) {
    auto ctx = x->ctx.lock();
    if (!ctx) throw std::runtime_error("Expert: ctx expired");
    // Diagnostics
    auto print_rank_dims = [&](const char* tag, const TensorPtr& T){
        if (!T) { std::cout << tag << "=(null)" << std::endl; return; }
        std::cout << tag << ": rank=" << T->shape.size() << " dims=";
        for (size_t di=0; di<T->shape.size(); ++di) {
            std::cout << T->shape[di] << (di+1<T->shape.size()?",":"");
        }
        std::cout << std::endl;
    };

    if (ow_verbose_moe()) {
        print_rank_dims("[MoE] x", x);
        print_rank_dims("[MoE] gate_proj", gate_proj);
        print_rank_dims("[MoE] up_proj", up_proj);
        print_rank_dims("[MoE] down_proj", down_proj);
    }
    
    // SwiGLU: (SiLU(gate) * up) -> down
    TensorPtr inter;
    if (gate_proj) {
        auto gate = Tensor::matmul_cache_friendly(x, gate_proj);
    auto up = Tensor::matmul_cache_friendly(x, up_proj);
        // sanitize
        size_t n_g = gate->nelements();
        for (size_t i = 0; i < n_g; ++i) {
            float v = gate->get_as_float_flat(i);
            if (!std::isfinite(v)) gate->set_from_float_flat(i, 0.0f);
        }
        size_t n_u = up->nelements();
        for (size_t i = 0; i < n_u; ++i) {
            float v = up->get_as_float_flat(i);
            if (!std::isfinite(v)) up->set_from_float_flat(i, 0.0f);
        }
        // SiLU(gate)
        auto act = Tensor::create(ctx, gate->shape, DType::FLOAT32);
        for (size_t i = 0; i < n_g; ++i) {
            float v = gate->get_as_float_flat(i);
            float sig = 1.0f / (1.0f + std::exp(-v));
            act->set_from_float_flat(i, v * sig);
        }
        // Elementwise mul
        inter = Tensor::create(ctx, up->shape, DType::FLOAT32);
        size_t n_i = inter->nelements();
        for (size_t i = 0; i < n_i; ++i) {
            inter->set_from_float_flat(i, act->get_as_float_flat(i) * up->get_as_float_flat(i));
        }
    } else {
        auto up = Tensor::matmul_cache_friendly(x, up_proj);
        size_t n_u = up->nelements();
        // sanitize and apply SiLU
        for (size_t i = 0; i < n_u; ++i) {
            float v = up->get_as_float_flat(i);
            if (!std::isfinite(v)) v = 0.0f;
            float sig = 1.0f / (1.0f + std::exp(-v));
            up->set_from_float_flat(i, v * sig);
        }
        inter = up;
    }
    auto down = Tensor::matmul_cache_friendly(inter, down_proj);
    return down;
}

TensorPtr SparseMoEBlock::forward(const TensorPtr& x) {
    auto ctx = x->ctx.lock();
    if (!ctx) throw std::runtime_error("SparseMoEBlock: ctx expired");
    // Router scores: [seq_len, num_experts_from_gate]
    auto scores = Tensor::matmul_cache_friendly(x, gate);
    int seq_len = x->shape[0];
    int hidden_size = x->shape[1];
    int gate_cols = scores->shape[1];
    int available = std::min(num_experts, gate_cols);
    auto output = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);

    // Top-1 routing per token for simplicity
    for (int i = 0; i < seq_len; ++i) {
        int best = 0;
        float best_score = -1e30f;
        for (int e = 0; e < available; ++e) {
            float s = scores->get_as_float_flat((size_t)i * gate_cols + e);
            if (!std::isfinite(s)) s = -1e30f;
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

// DecoderLayer forward with MRoPE-Interleave support
TensorPtr DecoderLayer::forward(const TensorPtr& hidden_states, int seq_len, const TensorPtr& position_ids) {
    auto ctx = hidden_states->ctx.lock();
    if (!ctx) throw std::runtime_error("DecoderLayer: ctx expired");
    
    // Pre-attention norm
    auto h1 = input_layernorm->forward(hidden_states);
    // Self attention with MRoPE-Interleave
    auto attn = self_attn->forward(h1, seq_len, position_ids);
    
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
    // Initialize KV caches for each layer to persist across forwards
    auto ctx = embed_tokens->ctx.lock();
    if (ctx) {
        for (auto &layer : layers) {
            if (layer && layer->self_attn) {
                layer->self_attn->init_cache(ctx, rotary_emb->max_seq_len);
            }
        }
    }
}

TensorPtr Qwen3VLTextModel::forward(const std::vector<int>& input_ids) {
    auto ctx = embed_tokens->ctx.lock();
    if (!ctx) throw std::runtime_error("Qwen3VLTextModel: ctx expired");
    int seq_len = (int)input_ids.size();

    // Pre-allocate result before mark so it survives release_to(mark)
    auto result = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);

    // Base mark for the whole forward; reclaim everything allocated after it at the end
    size_t mark = ctx->mark();

    // embedding lookup into hidden_states [seq_len, hidden_size]
    auto hidden_states = Tensor::create(ctx, std::vector<int>{seq_len, hidden_size}, DType::FLOAT32);
    int r0 = embed_tokens->shape[0];
    int r1 = embed_tokens->shape[1];
    if (r1 == hidden_size) {
        for (int i = 0; i < seq_len; ++i) {
            int tok = input_ids[i];
            if (tok < 0 || tok >= r0) tok = 0;
            for (int h = 0; h < hidden_size; ++h) {
                hidden_states->set_from_float_flat((size_t)i * hidden_size + h,
                    embed_tokens->get_as_float_flat((size_t)tok * hidden_size + h));
            }
        }
    } else if (r0 == hidden_size) {
        for (int i = 0; i < seq_len; ++i) {
            int tok = input_ids[i];
            if (tok < 0 || tok >= r1) tok = 0;
            for (int h = 0; h < hidden_size; ++h) {
                hidden_states->set_from_float_flat((size_t)i * hidden_size + h,
                    embed_tokens->get_as_float_flat((size_t)h * r1 + tok));
            }
        }
    } else {
        throw std::runtime_error("Qwen3VLTextModel: cannot infer hidden_size from embed_tokens shape");
    }

    // Sanitize hidden_states to avoid propagating NaN/Inf from FP8 embeddings
    for (size_t i = 0; i < hidden_states->nelements(); ++i) {
        float v = hidden_states->get_as_float_flat(i);
        if (!std::isfinite(v)) hidden_states->set_from_float_flat(i, 0.0f);
    }

    // Allocate ping-pong buffers for per-layer outputs (kept across per-layer releases)
    auto ping = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    auto pong = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    bool use_ping = true;
    
    // decoder layers with per-layer mark/release
    for (auto& layer : layers) {
        size_t lmark = ctx->mark();
        auto out_tmp = layer->forward(hidden_states, seq_len);
        auto target = use_ping ? ping : pong;
        // copy layer output into ping/pong buffer
        for (size_t i = 0; i < out_tmp->nelements(); ++i) {
            target->set_from_float_flat(i, out_tmp->get_as_float_flat(i));
        }
        ctx->release_to(lmark);
        hidden_states = target;
        use_ping = !use_ping;
    }
    
    // Apply final RMSNorm before output
    hidden_states = norm->forward(hidden_states);
    
    // Copy normalized hidden_states into result, then release temporaries
    for (size_t i = 0; i < hidden_states->nelements(); ++i) {
        result->set_from_float_flat(i, hidden_states->get_as_float_flat(i));
    }
    ctx->release_to(mark);
    return result;
}

// Qwen3VLTextModel forward with MRoPE-Interleave support
TensorPtr Qwen3VLTextModel::forward(const std::vector<int>& input_ids, const TensorPtr& position_ids) {
    auto ctx = embed_tokens->ctx.lock();
    if (!ctx) throw std::runtime_error("Qwen3VLTextModel: ctx expired");
    int seq_len = (int)input_ids.size();

    // Pre-allocate result before mark so it survives release_to(mark)
    auto result = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);

    // Base mark for the whole forward; reclaim everything allocated after it at the end
    size_t mark = ctx->mark();

    // embedding lookup into hidden_states [seq_len, hidden_size]
    auto hidden_states = Tensor::create(ctx, std::vector<int>{seq_len, hidden_size}, DType::FLOAT32);
    int r0 = embed_tokens->shape[0];
    int r1 = embed_tokens->shape[1];
    if (r1 == hidden_size) {
        for (int i = 0; i < seq_len; ++i) {
            int tok = input_ids[i];
            if (tok < 0 || tok >= r0) tok = 0;
            for (int h = 0; h < hidden_size; ++h) {
                hidden_states->set_from_float_flat((size_t)i * hidden_size + h,
                    embed_tokens->get_as_float_flat((size_t)tok * hidden_size + h));
            }
        }
    } else if (r0 == hidden_size) {
        for (int i = 0; i < seq_len; ++i) {
            int tok = input_ids[i];
            if (tok < 0 || tok >= r1) tok = 0;
            for (int h = 0; h < hidden_size; ++h) {
                hidden_states->set_from_float_flat((size_t)i * hidden_size + h,
                    embed_tokens->get_as_float_flat((size_t)h * r1 + tok));
            }
        }
    } else {
        throw std::runtime_error("Qwen3VLTextModel: cannot infer hidden_size from embed_tokens shape");
    }

    // Sanitize hidden_states to avoid propagating NaN/Inf from FP8 embeddings
    for (size_t i = 0; i < hidden_states->nelements(); ++i) {
        float v = hidden_states->get_as_float_flat(i);
        if (!std::isfinite(v)) hidden_states->set_from_float_flat(i, 0.0f);
    }

    // Allocate ping-pong buffers for per-layer outputs (kept across per-layer releases)
    auto ping = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    auto pong = Tensor::create(ctx, {seq_len, hidden_size}, DType::FLOAT32);
    bool use_ping = true;
    
    // decoder layers with per-layer mark/release and MRoPE-Interleave
    for (auto& layer : layers) {
        size_t lmark = ctx->mark();
        auto out_tmp = layer->forward(hidden_states, seq_len, position_ids);
        auto target = use_ping ? ping : pong;
        // copy layer output into ping/pong buffer
        for (size_t i = 0; i < out_tmp->nelements(); ++i) {
            target->set_from_float_flat(i, out_tmp->get_as_float_flat(i));
        }
        ctx->release_to(lmark);
        hidden_states = target;
        use_ping = !use_ping;
    }
    
    // Apply final RMSNorm before output
    hidden_states = norm->forward(hidden_states);
    
    // Copy normalized hidden_states into result, then release temporaries
    for (size_t i = 0; i < hidden_states->nelements(); ++i) {
        result->set_from_float_flat(i, hidden_states->get_as_float_flat(i));
    }
    ctx->release_to(mark);
    return result;
}

// WeightLoader implementation
std::shared_ptr<Qwen3VLTextModel> WeightLoader::build_model() {
    // Model configuration based on Qwen3VL config.json
    const int vocab_size = 151936;
    int num_layers = 48;     // default; will attempt to detect from weights
    // duplicate removed: num_heads
// duplicate removed: num_kv_heads
// duplicate removed: intermediate_size
// duplicate removed: num_experts
// duplicate removed: top_k
    
    // Auto-detect layer count from available weight keys (robust across naming schemes)
    int max_layer_index = -1;
    for (const auto& kv : weights) {
        const std::string& name = kv.first;
        auto extract = [&](const std::string& prefix){
            if (name.rfind(prefix, 0) == 0) {
                size_t pos = prefix.size();
                size_t end = name.find('.', pos);
                if (end != std::string::npos) {
                    int idx = std::atoi(name.substr(pos, end - pos).c_str());
                    if (idx > max_layer_index) max_layer_index = idx;
                }
            }
        };
        extract("model.language_model.layers.");
        extract("language_model.layers.");
        extract("model.layers.");
    }
    if (max_layer_index >= 0) num_layers = max_layer_index + 1;
    int num_heads = 32;
    int num_kv_heads = 4;

    // Infer head_dim/num_heads/num_kv_heads from first available layer weights
    int inferred_head_dim = -1;
    bool inferred_ok = false;
    
    // Get embedding weights (robust to different prefixes)
    auto embed_tokens = get_weight("model.embed_tokens.weight");
    if (!embed_tokens) embed_tokens = get_weight("model.language_model.embed_tokens.weight");
    if (!embed_tokens) embed_tokens = get_weight("language_model.embed_tokens.weight");
    if (!embed_tokens) embed_tokens = get_weight("embed_tokens.weight");
    if (!embed_tokens) embed_tokens = get_weight("wte.weight");
    if (!embed_tokens) {
        throw std::runtime_error("WeightLoader: embed_tokens.weight not found");
    }
    
    // Infer hidden_size from embedding weight shape to stay consistent with projections
    int hidden_size = (embed_tokens->shape.size() == 2) ? embed_tokens->shape[1] : 0;
    if (hidden_size <= 0) {
        throw std::runtime_error("WeightLoader: invalid hidden_size inferred from embed_tokens");
    }
    std::cout << "[CFG] hidden_size=" << hidden_size << std::endl;

    for (int li = 0; li < num_layers && !inferred_ok; ++li) {
        std::vector<std::string> layer_prefixes = {
            std::string("model.layers.") + std::to_string(li) + ".",
            std::string("model.language_model.layers.") + std::to_string(li) + ".",
            std::string("language_model.layers.") + std::to_string(li) + "."
        };
        TensorPtr q_proj_raw_probe, k_proj_raw_probe, v_proj_raw_probe, q_norm_weight_probe;
        for (const auto &lp : layer_prefixes) {
            if (!q_proj_raw_probe) q_proj_raw_probe = get_weight(lp + "self_attn.q_proj.weight");
            if (!k_proj_raw_probe) k_proj_raw_probe = get_weight(lp + "self_attn.k_proj.weight");
            if (!v_proj_raw_probe) v_proj_raw_probe = get_weight(lp + "self_attn.v_proj.weight");
            if (!q_norm_weight_probe) q_norm_weight_probe = get_weight(lp + "self_attn.q_norm.weight");
        }
        if (q_proj_raw_probe && k_proj_raw_probe && v_proj_raw_probe && q_norm_weight_probe) {
            auto q_proj_oriented = ensure_linear_weight_or_transpose(q_proj_raw_probe, hidden_size);
            auto k_proj_oriented = ensure_linear_weight_or_transpose(k_proj_raw_probe, hidden_size);
            auto v_proj_oriented = ensure_linear_weight_or_transpose(v_proj_raw_probe, hidden_size);
            int q_out = (q_proj_oriented->shape.size()==2) ? q_proj_oriented->shape[1] : -1;
            int k_out = (k_proj_oriented->shape.size()==2) ? k_proj_oriented->shape[1] : -1;
            int v_out = (v_proj_oriented->shape.size()==2) ? v_proj_oriented->shape[1] : -1;
            int hd_from_norm = (int)q_norm_weight_probe->nelements();
            if (hd_from_norm > 0) inferred_head_dim = hd_from_norm;
            if (inferred_head_dim > 0) {
                int heads_candidate = hidden_size / inferred_head_dim;
                int kv_heads_candidate = (k_out > 0) ? (k_out / inferred_head_dim) : -1;
                if (heads_candidate > 0 && kv_heads_candidate > 0) {
                    num_heads = heads_candidate;
                    num_kv_heads = kv_heads_candidate;
                    inferred_ok = true;
                    std::cout << "[CFG] inferred num_heads=" << num_heads
                              << " num_kv_heads=" << num_kv_heads
                              << " head_dim=" << inferred_head_dim
                              << " (q_out=" << q_out << ", k_out=" << k_out << ", v_out=" << v_out << ")" << std::endl;
                }
            }
        }
    }

    int intermediate_size = -1;  // infer from weights when available
    const int num_experts = 128;
    const int top_k = 8;
    
    // Build decoder layers
    std::vector<std::shared_ptr<DecoderLayer>> layers;
    
    for (int i = 0; i < num_layers; ++i) {
        // Try multiple possible layer prefixes
        std::vector<std::string> layer_prefixes = {
            std::string("model.layers.") + std::to_string(i) + ".",
            std::string("model.language_model.layers.") + std::to_string(i) + ".",
            std::string("language_model.layers.") + std::to_string(i) + "."
        };
        
        // Attention weights
        TensorPtr q_proj_raw, k_proj_raw, v_proj_raw, o_proj_raw, q_norm_weight, k_norm_weight;
        for (const auto &lp : layer_prefixes) {
            if (!q_proj_raw) q_proj_raw = get_weight(lp + "self_attn.q_proj.weight");
            if (!k_proj_raw) k_proj_raw = get_weight(lp + "self_attn.k_proj.weight");
            if (!v_proj_raw) v_proj_raw = get_weight(lp + "self_attn.v_proj.weight");
            if (!o_proj_raw) o_proj_raw = get_weight(lp + "self_attn.o_proj.weight");
            if (!q_norm_weight) q_norm_weight = get_weight(lp + "self_attn.q_norm.weight");
            if (!k_norm_weight) k_norm_weight = get_weight(lp + "self_attn.k_norm.weight");
        }
        
        if (!q_proj_raw || !k_proj_raw || !v_proj_raw || !o_proj_raw || !q_norm_weight || !k_norm_weight) {
            std::cerr << "Warning: Missing attention weights for layer " << i << std::endl;
            continue;
        }
        // Fix orientation to [hidden_size, out_dim]
        auto q_proj = ensure_linear_weight_or_transpose(q_proj_raw, hidden_size, hidden_size);
        auto k_proj = ensure_linear_weight_or_transpose(k_proj_raw, hidden_size);
        auto v_proj = ensure_linear_weight_or_transpose(v_proj_raw, hidden_size);
        auto o_proj = ensure_linear_weight_or_transpose(o_proj_raw, hidden_size, hidden_size);
        // Avoid FP8 NaNs in output projection by converting to FLOAT32
        o_proj = o_proj->astype(DType::FLOAT32);
         
        // Create attention layer
        auto attention = std::make_shared<MultiHeadAttention>(
            q_proj, k_proj, v_proj, o_proj,
            q_norm_weight, k_norm_weight,
            num_heads, num_kv_heads, hidden_size
        );
        
        // MoE weights - merged expert weights
        TensorPtr gate_weight_raw, experts_gate_up_proj, experts_down_proj;
        for (const auto &lp : layer_prefixes) {
            if (!gate_weight_raw) gate_weight_raw = get_weight(lp + "mlp.gate.weight");
            if (!experts_gate_up_proj) experts_gate_up_proj = get_weight(lp + "mlp.experts.gate_up_proj");
            if (!experts_down_proj) experts_down_proj = get_weight(lp + "mlp.experts.down_proj");
        }
        
        // Decide MoE vs Dense MLP
        std::shared_ptr<SparseMoEBlock> moe;
        std::vector<std::shared_ptr<Expert>> experts;
        if (!gate_weight_raw || !experts_gate_up_proj || !experts_down_proj) {
            // Dense MLP fallback: Qwen uses SwiGLU with gate_proj/up_proj/down_proj
            TensorPtr dense_gate_proj_raw, dense_up_proj_raw, dense_down_proj_raw;
            for (const auto &lp : layer_prefixes) {
                if (!dense_gate_proj_raw) dense_gate_proj_raw = get_weight(lp + "mlp.gate_proj.weight");
                if (!dense_up_proj_raw) dense_up_proj_raw = get_weight(lp + "mlp.up_proj.weight");
                if (!dense_down_proj_raw) dense_down_proj_raw = get_weight(lp + "mlp.down_proj.weight");
            }
            if (!dense_up_proj_raw || !dense_down_proj_raw) {
                std::cerr << "Warning: Missing MLP weights for layer " << i << std::endl;
                continue;
            }
            auto gate_proj = dense_gate_proj_raw ? ensure_linear_weight_or_transpose(dense_gate_proj_raw, hidden_size) : nullptr;
            auto up_proj = ensure_linear_weight_or_transpose(dense_up_proj_raw, hidden_size);
            int I = up_proj->shape.size() == 2 ? up_proj->shape[1] : -1;
            if (I <= 0) {
                std::cerr << "Warning: Invalid up_proj dims for layer " << i << std::endl;
                continue;
            }
            auto down_proj = ensure_linear_weight_or_transpose(dense_down_proj_raw, I, hidden_size);
            experts.push_back(std::make_shared<Expert>(gate_proj, up_proj, down_proj));
            // Stub router gate [hidden_size,1] -> always route to expert 0
            auto ctx = embed_tokens->ctx.lock();
            auto gate_stub = ow::nn::Tensor::create(ctx, std::vector<int>{hidden_size, 1}, DType::FLOAT32);
            for (size_t gi = 0; gi < gate_stub->nelements(); ++gi) gate_stub->set_from_float_flat(gi, 0.0f);
            moe = std::make_shared<SparseMoEBlock>(gate_stub, experts, 1);
            std::cout << "[MLP] Using dense SwiGLU MLP for layer " << i << std::endl;
        } else {
            auto gate_weight = ensure_linear_weight_or_transpose(gate_weight_raw, hidden_size);
            // Derive router expert count from gate weight dims (supports transposed layout)
            int router_experts = (gate_weight->shape[0] == hidden_size)
                                   ? gate_weight->shape[1]
                                   : gate_weight->shape[0];
            if (ow_verbose_moe()) {
                std::cout << "[MoE] Router experts from gate weight: " << router_experts << std::endl;
            }
            
            // Prepare expert container
            if (experts_gate_up_proj && experts_down_proj &&
                experts_gate_up_proj->shape.size() == 3 && experts_down_proj->shape.size() == 3) {
                int E = experts_gate_up_proj->shape[0];
                int H = experts_gate_up_proj->shape[1];
                int GU = experts_gate_up_proj->shape[2]; // gate+up merged
                int I_down = experts_down_proj->shape[1];
                int H_down = experts_down_proj->shape[2];
                int I = I_down;
                if (I <= 0 || (GU % 2) != 0) I = GU / 2;
                intermediate_size = (intermediate_size < 0) ? I : intermediate_size;
                int build_E = std::min(E, router_experts);

                if (ow_verbose_moe()) {
                    std::cout << "[MoE] Slicing merged experts: E=" << E
                              << " H=" << H << " GU=" << GU
                              << " -> inferred I=" << I << " H_down=" << H_down
                              << " build_E=" << build_E << std::endl;
                }
                for (int e = 0; e < build_E; ++e) {
                    auto gate_e_3d = experts_gate_up_proj->slice_view({e, 0, 0}, {1, H, I});
                    auto gate_e = gate_e_3d->view({H, I}, {gate_e_3d->strides[1], gate_e_3d->strides[2]});
                    gate_e = ensure_linear_weight_or_transpose(gate_e, hidden_size, I);
                    auto up_e_3d = experts_gate_up_proj->slice_view({e, 0, I}, {1, H, I});
                    auto up_e = up_e_3d->view({H, I}, {up_e_3d->strides[1], up_e_3d->strides[2]});
                    up_e = ensure_linear_weight_or_transpose(up_e, hidden_size, I);
                    auto down_e_3d = experts_down_proj->slice_view({e, 0, 0}, {1, I, hidden_size});
                    auto down_e = down_e_3d->reshape_view({I, hidden_size});
                    down_e = ensure_linear_weight_or_transpose(down_e, I, hidden_size);
                    experts.push_back(std::make_shared<Expert>(gate_e, up_e, down_e));
                }

                if (ow_verbose_moe()) {
                    std::cout << "[MoE] Built " << experts.size() << " experts with 2D weights (gate/up/down)" << std::endl;
                }
            } else {
                int I = -1;
                if (experts_gate_up_proj && experts_gate_up_proj->shape.size() == 2) {
                    auto gu_oriented = ensure_linear_weight_or_transpose(experts_gate_up_proj, hidden_size);
                    int GU = gu_oriented->shape[1];
                    I = GU / 2;
                    auto gate_e = gu_oriented->slice_view({0, 0}, {hidden_size, I});
                    auto up_e = gu_oriented->slice_view({0, I}, {hidden_size, I});
                    gate_e = ensure_linear_weight_or_transpose(gate_e, hidden_size, I);
                    up_e = ensure_linear_weight_or_transpose(up_e, hidden_size, I);
                    auto down2d = ensure_linear_weight_or_transpose(experts_down_proj, I, hidden_size);
                    experts.push_back(std::make_shared<Expert>(gate_e, up_e, down2d));
                } else {
                    auto up2d = ensure_linear_weight_or_transpose(experts_gate_up_proj, hidden_size);
                    auto down2d = ensure_linear_weight_or_transpose(experts_down_proj, hidden_size);
                    experts.push_back(std::make_shared<Expert>(nullptr, up2d, down2d));
                }

                if (ow_verbose_moe()) {
                    std::cout << "[MoE] Built merged single expert (fallback)" << std::endl;
                }
            }
            moe = std::make_shared<SparseMoEBlock>(gate_weight, experts, top_k);
        }

        // Layer norms
        TensorPtr input_layernorm_weight, post_attention_layernorm_weight;
        for (const auto &lp : layer_prefixes) {
            if (!input_layernorm_weight) input_layernorm_weight = get_weight(lp + "input_layernorm.weight");
            if (!post_attention_layernorm_weight) post_attention_layernorm_weight = get_weight(lp + "post_attention_layernorm.weight");
        }
        
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
    
    // Final layer norm (robust to different prefixes)
    auto norm_weight = get_weight("model.norm.weight");
    if (!norm_weight) norm_weight = get_weight("model.language_model.norm.weight");
    if (!norm_weight) norm_weight = get_weight("language_model.norm.weight");
    if (!norm_weight) norm_weight = get_weight("norm.weight");
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

