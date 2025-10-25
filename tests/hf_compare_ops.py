#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成标准化算子测试向量，并将其保存为自定义 .tensor 二进制格式：
[int32 rank][int32 dims...][float32 data...]

涵盖：
- 线性层 (X, W, b, Y)
- 激活函数 (relu/sigmoid/tanh/gelu_approx/gelu_exact/silu/softmax)
- RMSNorm (Qwen2/LlamaRMSNorm 等价)
- RoPE (默认公式生成 cos/sin)
- 注意力机制（QKV 线性、RoPE、SDPA 输出）

对比来源：HuggingFace Transformers + PyTorch
"""
import os
import struct
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def save_tensor_bin(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    rank = array.ndim
    dims = list(array.shape)
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', rank))
        for d in dims:
            f.write(struct.pack('<i', int(d)))
        f.write(array.astype(np.float32).tobytes(order='C'))


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def gen_linear(out_dir: Path, m=3, k=5, n=4, seed=123):
    set_seed(seed)
    X = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(k, n, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32)
    Y = X @ W + b
    save_tensor_bin(out_dir / 'linear_X.tensor', X.numpy())
    save_tensor_bin(out_dir / 'linear_W.tensor', W.numpy())
    save_tensor_bin(out_dir / 'linear_b.tensor', b.numpy())
    save_tensor_bin(out_dir / 'linear_Y.tensor', Y.numpy())


def gen_activations(out_dir: Path, m=2, n=7, seed=21):
    set_seed(seed)
    X = torch.randn(m, n, dtype=torch.float32)
    save_tensor_bin(out_dir / 'act_input.tensor', X.numpy())
    # relu/sigmoid/tanh
    save_tensor_bin(out_dir / 'relu.tensor', F.relu(X).numpy())
    save_tensor_bin(out_dir / 'sigmoid.tensor', torch.sigmoid(X).numpy())
    save_tensor_bin(out_dir / 'tanh.tensor', torch.tanh(X).numpy())
    # gelu approx/exact
    gelu_exact = F.gelu(X, approximate='none')
    gelu_tanh = F.gelu(X, approximate='tanh')
    save_tensor_bin(out_dir / 'gelu_exact.tensor', gelu_exact.numpy())
    save_tensor_bin(out_dir / 'gelu_tanh.tensor', gelu_tanh.numpy())
    # silu
    silu = F.silu(X)
    save_tensor_bin(out_dir / 'silu.tensor', silu.numpy())
    # softmax last-dim
    soft = F.softmax(X, dim=-1)
    save_tensor_bin(out_dir / 'softmax.tensor', soft.numpy())


def gen_rmsnorm(out_dir: Path, hidden_size=16, batch=3, seed=7, eps=1e-6):
    set_seed(seed)
    X = torch.randn(batch, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    var = (X.pow(2).mean(dim=-1, keepdim=True) + eps)
    Xnorm = X * torch.rsqrt(var)
    Y = Xnorm * weight
    save_tensor_bin(out_dir / 'rms_X.tensor', X.numpy())
    save_tensor_bin(out_dir / 'rms_weight.tensor', weight.numpy())
    save_tensor_bin(out_dir / 'rms_Y.tensor', Y.numpy())


def default_rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0):
    assert head_dim % 2 == 0, 'head_dim must be even'
    # freqs: [S, D/2]
    i = torch.arange(0, head_dim // 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (2.0 * i / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [S, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, D]
    cos = torch.cos(emb)
    sin = torch.sin(emb)
    # 返回 [S, 1, D] 以便与 [B,S,H,D] 广播
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return cos, sin


def gen_rope(out_dir: Path, head_dim=16, max_pos=64, seq_len=8, batch=2, seed=11):
    set_seed(seed)
    cos, sin = default_rope_cos_sin(seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    save_tensor_bin(out_dir / 'rope_cos.tensor', cos.numpy())
    save_tensor_bin(out_dir / 'rope_sin.tensor', sin.numpy())
    save_tensor_bin(out_dir / 'rope_pos_ids.tensor', position_ids.numpy().astype(np.float32))


def gen_attention(out_dir: Path, B=1, S=4, H=2, D=8, seed=99):
    set_seed(seed)
    hidden_size = H * D
    X = torch.randn(B, S, hidden_size, dtype=torch.float32)
    Wq = torch.randn(hidden_size, H * D, dtype=torch.float32)
    Wk = torch.randn(hidden_size, H * D, dtype=torch.float32)
    Wv = torch.randn(hidden_size, H * D, dtype=torch.float32)
    Wo = torch.randn(H * D, hidden_size, dtype=torch.float32)
    bq = torch.randn(H * D, dtype=torch.float32)
    bk = torch.randn(H * D, dtype=torch.float32)
    bv = torch.randn(H * D, dtype=torch.float32)
    bo = torch.randn(hidden_size, dtype=torch.float32)

    # Q,K,V
    Q = X @ Wq + bq
    K = X @ Wk + bk
    V = X @ Wv + bv

    # 重新整理维度为 [B, S, H, D]
    def reshape_bshd(T):
        return T.view(B, S, H, D)
    Qh = reshape_bshd(Q)
    Kh = reshape_bshd(K)
    Vh = reshape_bshd(V)

    # RoPE 缓存，生成 [S, 1, D] 形状
    cos, sin = default_rope_cos_sin(S, D)
    pos_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).repeat(B, 1)

    # RMSNorm + scaling，与 backend 实现对齐
    eps = 1e-6
    def rms_norm(x):
        var = (x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * torch.rsqrt(var)
    Qh = rms_norm(Qh); Kh = rms_norm(Kh)
    Qh = Qh * (1.0 / np.sqrt(D))

    # 应用 RoPE（与 utils.apply_rotary_pos_emb 逻辑一致）
    def rotate_half(x):
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(q, k, cos, sin):
        # cos,sin: [S, 1, D]，广播到 [B, S, H, D]
        cos_b = cos.expand(B, S, 1, D)
        sin_b = sin.expand(B, S, 1, D)
        q_embed = q * cos_b + rotate_half(q) * sin_b
        k_embed = k * cos_b + rotate_half(k) * sin_b
        return q_embed, k_embed

    Qr, Kr = apply_rope(Qh, Kh, cos, sin)

    # Scaled dot-product attention with causal mask
    scores = torch.einsum('bshd,bthd->bhst', Qr, Kr)  # [B, H, S, S]
    causal = torch.full_like(scores, -1e9)
    idx_i = torch.arange(S).view(1, 1, S, 1)
    idx_j = torch.arange(S).view(1, 1, 1, S)
    mask = (idx_j <= idx_i)
    scores = torch.where(mask, scores, causal)
    attn = torch.softmax(scores, dim=-1)
    O_heads = torch.einsum('bhst,bthd->bshd', attn, Vh)  # [B, S, H, D]
    O = O_heads.reshape(B, S, H * D) @ Wo + bo

    # 保存张量
    save_tensor_bin(out_dir / 'attn_X.tensor', X.numpy())
    save_tensor_bin(out_dir / 'attn_Wq.tensor', Wq.numpy())
    save_tensor_bin(out_dir / 'attn_Wk.tensor', Wk.numpy())
    save_tensor_bin(out_dir / 'attn_Wv.tensor', Wv.numpy())
    save_tensor_bin(out_dir / 'attn_Wo.tensor', Wo.numpy())
    save_tensor_bin(out_dir / 'attn_bq.tensor', bq.numpy())
    save_tensor_bin(out_dir / 'attn_bk.tensor', bk.numpy())
    save_tensor_bin(out_dir / 'attn_bv.tensor', bv.numpy())
    save_tensor_bin(out_dir / 'attn_bo.tensor', bo.numpy())
    save_tensor_bin(out_dir / 'attn_pos_ids.tensor', pos_ids.numpy().astype(np.float32))
    save_tensor_bin(out_dir / 'attn_cos.tensor', cos.numpy())
    save_tensor_bin(out_dir / 'attn_sin.tensor', sin.numpy())
    save_tensor_bin(out_dir / 'attn_Y.tensor', O.detach().numpy())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='tests/data', help='输出目录')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    out_dir = Path(args.out)

    print('[HF] 生成线性层测试向量...')
    gen_linear(out_dir / 'linear', seed=args.seed)

    print('[HF] 生成激活函数测试向量...')
    gen_activations(out_dir / 'activations', seed=args.seed + 1)

    print('[HF] 生成 RMSNorm 测试向量...')
    gen_rmsnorm(out_dir / 'rmsnorm', seed=args.seed + 2)

    print('[HF] 生成 RoPE 测试向量...')
    gen_rope(out_dir / 'rope', seed=args.seed + 3)

    print('[HF] 生成注意力测试向量...')
    gen_attention(out_dir / 'attention', seed=args.seed + 4)

    print('完成。数据保存到', str(out_dir))


if __name__ == '__main__':
    main()