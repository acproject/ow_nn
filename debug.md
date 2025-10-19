建议修复

- 应用最终 RMSNorm：
  
  - 在 Qwen3VLTextModel::forward() 的输出前调用一次 norm->forward(hidden_states) ，再进行拷贝并 release_to(mark) 。这样隐藏状态会被幅度归一化，logits 不会普遍冲到裁剪上限。
  - 位置建议：就在你 “final norm” 注释那里，插入 hidden_states = norm->forward(hidden_states); 。
- 去掉或放宽 logits 人为裁剪：
  
  - matvec_rows_dot 里目前将最终结果裁到 [-50, 50] ； argmax_logits 里也重复做了裁剪。建议删除这两处“最终裁剪”，保留对非有限值的过滤即可。Top-k 调试就能看到真实差异，不会一堆 50。
  - 如果担心 softmax 溢出，做法是对 softmax 前先减去最大值（你在注意力里已这么做），而不是硬裁剪 logits。
- 使用更可靠的 LMHead 点积实现：
  
  - 直接改用 Tensor::matvec_blocked_mt(last_hidden, vocab_weight) ，它支持 [K,N] 或 [N,K] 两种排布，内部有分块并行、向量化和更稳健的数值处理。你当前 vocab_weight 为 [V,K] ，符合它的 [N,K] 分支，无需转置。
  - 这能显著提升性能，还减少你在 matvec_rows_dot 中自行做的“自定义裁剪”副作用。
- 可选的数值与权重处理：
  
  - 将 lm_head.weight 同样转成 FLOAT32 （与 o_proj 一致），降低 BF16 量化误差在点积上的放大效应。
  - 在打印 top-k 时，不要再对值做裁剪，只做 isfinite 检查，这样调试更真实。
参考修改示例

- 在 Qwen3VLTextModel::forward() 尾部：
  
  - 原来：
    - “final norm” 注释后直接拷贝 hidden_states 到 result 。
  - 改为：
    - hidden_states = norm->forward(hidden_states);
    - 再拷贝到 result ，释放到 mark 。
- 在生成阶段的 logits 计算：
  
  - 将
    - auto logits = matvec_rows_dot(last_hidden, vocab_weight);
  - 改为
    - auto logits = ow::nn::Tensor::matvec_blocked_mt(last_hidden, vocab_weight);
  - 并在 argmax_logits 中移除最终的 [-50, 50] 裁剪。
内存建议（对齐 ggml 风格）

- 分离权重和计算内存：
  - 权重保持内存映射或常驻（你已在 MemoryOptimizer 里做了），计算使用 Context 的 scratch 区域。
- 分层释放（mark/release 粒度下沉）：
  - 当前 Qwen3VLTextModel::forward() 在所有层运行完才 release_to(mark) ，导致每层的中间张量累积。
  - 建议在每一层内部或每层循环体使用 “双缓冲 + mark/release”的模式：
    - 在进入一层时 size_t lmark = ctx->mark();
    - 计算出该层的输出后，拷贝到一个预分配的 hidden_pingpong 缓冲（两份 [seq_len, hidden_size] 交替使用）。
    - ctx->release_to(lmark); 释放该层的中间张量。
    - 将 hidden_states 指向另一块 pingpong 缓冲并进入下一层。
  - 这样单层的临时内存能及时回收，峰值更像 ggml 的 ggml_allocr 方式。
- 引入 KV Cache（如果后续要做增量生成）：
  - 目前每步用整段 seq_len 计算，内存和算力都压得很重。实现 KV Cache 后，每步只计算新 token 的 Q/K/V 并复用历史 K/V，内存峰值和延迟都大幅下降，这是 ggml 系的常见策略。
- Scratch 打包已在你的 matmul_cache_friendly 中用到：
  - ctx->scratch_alloc_matrix_pack 的打包思路是对齐 ggml 的“面板打包 + 分块计算”。保持这一块，并减少不必要的张量物化。
快速验证建议

- 先只做两处小改动：1) 在 Qwen3VLTextModel::forward() 应用最终 norm ；2) 切换到 Tensor::matvec_blocked_mt 并去掉 logits 裁剪。
- 然后重新跑一遍生成，观察：
  - [DBG top-5] 是否不再都是 50，数值是否有明显区分。
  - 是否还会出现 NaN 警告（理应减少）。
  - 输出的片段是否更合理（字符/分词片段）。
如果你希望我直接为你补上最终 RMSNorm 和切换到 matvec_blocked_mt ，我可以马上打补丁；也可以照你的偏好调整裁剪阈值或打印更多调试信息（如 logits 的分布统计）来进一步定位.

---

KV Cache 实装小结（与 ggml 对齐）

- 注意力前向改动：
  - Q 仍按整段 `matmul_cache_friendly` 计算，随后 `q_norm`+RoPE。
  - K/V 采用“最后一行增量”：对新 token `x_i` 使用 `Tensor::matvec_blocked_mt(x_i, k_proj/v_proj)`，K 经 `k_norm` 后按位应用 RoPE，分别写入 `k_cache/v_cache`。
  - 计算注意力分数时，直接读取 `k_cache/v_cache`，仅遍历到 `i`（因因果掩码）。
- 缓存初始化与生命周期：
  - 在 `MultiHeadAttention::init_cache(ctx, max_seq_len)` 中分配 `k_cache/v_cache` 为 `{max_seq_len, num_kv_heads, head_dim}`，并在 `Qwen3VLTextModel` 构造阶段调用一次，保证不被 `mark/release` 回收。
  - 当 `seq_len < cache_len`（新提示词）时重置 `cache_len=0`，避免旧缓存污染。
- 计算与内存：
  - `matvec_blocked_mt` 内部沿用 scratch 的“面板打包 + 分块计算”，对齐 ggml 的 pack+block 风格，降低不必要的张量物化与峰值内存。
  - 每步仅新增一行 K/V，`k_cache/v_cache` 常驻，峰值随 `seq_len` 增长但不再重复分配整段 K/V。

LMHead 改动复盘

- 已将词表点积切换为 `Tensor::matvec_blocked_mt`，去除了 `argmax_logits` 的最终裁剪，仅保留非有限值过滤，便于真实观测 Top-K。
- 与最终 `RMSNorm` 配合，logits 幅度进入更合理范围，减少“顶到 50”现象。

后续观察点

- 生成阶段内存曲线应更平滑：新增 token 时内存主要增加在缓存行写入，不再重复构建整段 K/V。
- 若需进一步降峰值：可在解码器层内引入 ping-pong 缓冲 + 按层 `mark/release`，但这不影响 KV 常驻。