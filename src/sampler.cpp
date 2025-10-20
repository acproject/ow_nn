#include <algorithm>
#include <random>
#include <cmath>
#include <cstdint>
#include <vector>
#include "sampler.h"

// Simple single-sequence sampler implementation matching sampler.h
// Supports: temperature, top_p, top_k, repetition_penalty
// Does not support beam search; maintains internal last_token_ids for repetition penalty.

namespace {
static float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
}

void sampler::reset(size_t keep_last_n) {
    if (keep_last_n == 0 || last_token_ids.size() <= keep_last_n) return;
    last_token_ids.erase(last_token_ids.begin(), last_token_ids.end() - keep_last_n);
}

void sampler::softmax(float* logits, std::vector<std::vector<size_t>> /*picks*/, std::vector<uint32_t> /*max_indices*/) {
    // In-place softmax over vocab_size entries in logits
    // numerically stable: subtract max
    if (vocab_size == 0) return;
    float maxv = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab_size; ++i) {
        float v = logits[i];
        if (!std::isfinite(v)) continue;
        if (v > maxv) maxv = v;
    }
    if (!std::isfinite(maxv)) maxv = 0.f;
    double sum = 0.0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        float v = logits[i];
        if (!std::isfinite(v)) v = -1e9f;
        v = (v - maxv);
        logits[i] = v;
    }
    // temperature scaling
    float t = temperature;
    if (!std::isfinite(t) || t <= 1e-6f) t = 1.0f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        float v = logits[i] / t;
        logits[i] = v;
    }
    // exponentiate and sum
    for (uint32_t i = 0; i < vocab_size; ++i) {
        float v = logits[i];
        double ev = std::exp(static_cast<double>(v));
        if (!std::isfinite(ev)) ev = 0.0;
        logits[i] = static_cast<float>(ev);
        sum += ev;
    }
    if (sum <= 0.0) {
        float uniform = 1.0f / (float)(vocab_size > 0 ? vocab_size : 1);
        for (uint32_t i = 0; i < vocab_size; ++i) logits[i] = uniform;
        return;
    }
    for (uint32_t i = 0; i < vocab_size; ++i) logits[i] = static_cast<float>(logits[i] / sum);
}

static void apply_repetition_penalty(float* probs, uint32_t vocab_size, const std::vector<uint32_t>& last_ids, float penalty) {
    if (!std::isfinite(penalty) || penalty <= 1.0f || last_ids.empty()) return;
    // Build histogram of last ids
    std::vector<uint32_t> counts(vocab_size, 0);
    for (uint32_t id : last_ids) if (id < vocab_size) counts[id]++;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (counts[i] > 0) {
            probs[i] = probs[i] / penalty; // simple downweight
        }
    }
    // renormalize
    double sum = 0.0;
    for (uint32_t i = 0; i < vocab_size; ++i) sum += probs[i];
    if (sum > 0.0) {
        for (uint32_t i = 0; i < vocab_size; ++i) probs[i] = static_cast<float>(probs[i] / sum);
    }
}

static std::vector<uint32_t> topk_filter_indices(const float* probs, uint32_t vocab_size, uint32_t top_k) {
    if (top_k == 0 || top_k >= vocab_size) {
        std::vector<uint32_t> idx(vocab_size);
        for (uint32_t i = 0; i < vocab_size; ++i) idx[i] = i;
        return idx;
    }
    std::vector<std::pair<float,uint32_t>> pairs;
    pairs.reserve(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) pairs.emplace_back(probs[i], i);
    std::partial_sort(pairs.begin(), pairs.begin()+top_k, pairs.end(), [](auto &a, auto &b){ return a.first > b.first; });
    std::vector<uint32_t> idx; idx.reserve(top_k);
    for (uint32_t i = 0; i < top_k; ++i) idx.push_back(pairs[i].second);
    return idx;
}

static std::vector<uint32_t> topp_filter_indices(const float* probs, uint32_t vocab_size, float top_p) {
    float p = clamp01(top_p);
    if (p <= 0.f) {
        // only highest prob
        uint32_t maxi = 0; float maxv = probs[0];
        for (uint32_t i = 1; i < vocab_size; ++i) if (probs[i] > maxv) { maxv = probs[i]; maxi = i; }
        return {maxi};
    }
    if (p >= 0.9999f) {
        std::vector<uint32_t> all(vocab_size);
        for (uint32_t i = 0; i < vocab_size; ++i) all[i] = i;
        return all;
    }
    std::vector<std::pair<float,uint32_t>> pairs;
    pairs.reserve(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) pairs.emplace_back(probs[i], i);
    std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b){ return a.first > b.first; });
    double acc = 0.0;
    std::vector<uint32_t> idx;
    idx.reserve(vocab_size);
    for (auto &pr : pairs) {
        acc += pr.first;
        idx.push_back(pr.second);
        if (acc >= p) break;
    }
    if (idx.empty()) idx.push_back(pairs.front().second);
    return idx;
}

static uint32_t sample_from_indices(const float* probs, const std::vector<uint32_t>& idx, std::mt19937 &rng) {
    if (idx.empty()) return 0;
    // Build distribution over selected indices
    double sum = 0.0;
    for (uint32_t id : idx) sum += probs[id];
    if (sum <= 0.0) return idx.front();
    std::uniform_real_distribution<double> uni(0.0, sum);
    double r = uni(rng);
    double acc = 0.0;
    for (uint32_t id : idx) {
        acc += probs[id];
        if (r <= acc) return id;
    }
    return idx.back();
}

void sampler::topk(float* probs, std::vector<uint32_t>& out_indices) {
    out_indices = topk_filter_indices(probs, vocab_size, top_k);
}

void sampler::topp(float* probs, std::vector<uint32_t>& out_indices) {
    out_indices = topp_filter_indices(probs, vocab_size, top_p);
}

void sampler::max(float* probs, std::vector<uint32_t>& out_indices) {
    // Greedy: select argmax
    uint32_t maxi = 0; float maxv = probs[0];
    for (uint32_t i = 1; i < vocab_size; ++i) if (probs[i] > maxv) { maxv = probs[i]; maxi = i; }
    out_indices = {maxi};
}

void sampler::sample(float* logits, std::vector<uint32_t>& output_tokens) {
    if (vocab_size == 0) return;
    // Convert logits to probabilities
    std::vector<float> probs(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) probs[i] = logits[i];
    if (apply_softmax) softmax(probs.data(), {}, {});

    // repetition penalty
    apply_repetition_penalty(probs.data(), vocab_size, last_token_ids, repetition_penalty);

    // Filter candidates
    std::vector<uint32_t> idx_k, idx_p;
    if (top_k > 0) topk(probs.data(), idx_k); else { idx_k.resize(vocab_size); for (uint32_t i=0;i<vocab_size;++i) idx_k[i]=i; }
    if (top_p > 0.f) topp(probs.data(), idx_p); else { idx_p.resize(vocab_size); for (uint32_t i=0;i<vocab_size;++i) idx_p[i]=i; }

    // Intersect idx_k and idx_p to get final candidate set
    std::vector<uint32_t> candidate;
    candidate.reserve(std::min(idx_k.size(), idx_p.size()));
    std::sort(idx_k.begin(), idx_k.end());
    std::sort(idx_p.begin(), idx_p.end());
    std::set_intersection(idx_k.begin(), idx_k.end(), idx_p.begin(), idx_p.end(), std::back_inserter(candidate));
    if (candidate.empty()) candidate = idx_k.size() < idx_p.size() ? idx_k : idx_p;

    // Sample or greedy
    uint32_t chosen = 0;
    if (do_sample) {
        std::random_device rd;
        std::mt19937 rng(rd());
        chosen = sample_from_indices(probs.data(), candidate, rng);
    } else {
        // fallback greedy within candidate
        float best = -1.f; uint32_t best_id = candidate.front();
        for (uint32_t id : candidate) if (probs[id] > best) { best = probs[id]; best_id = id; }
        chosen = best_id;
    }

    output_tokens.push_back(chosen);
    last_token_ids.push_back(chosen);
    if (last_token_ids.size() > last_max_keep) {
        last_token_ids.erase(last_token_ids.begin());
    }
}