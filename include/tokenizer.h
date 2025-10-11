#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ow::nn {

// 简化版 BPE 分词器接口
class Tokenizer {
public:
  // 加载 vocab.json 和 merges.txt
  Tokenizer(const std::string &vocab_path, const std::string &merges_path);

  // 编码：text -> token ids
  std::vector<int> encode(const std::string &text);

  // 解码：ids -> text（尽力而为）
  std::string decode(const std::vector<int> &ids) const;

private:
  // pair<string,string> 的哈希
  struct PairHash {
    std::size_t
    operator()(const std::pair<std::string, std::string> &p) const noexcept {
      std::hash<std::string> h;
      std::size_t s1 = h(p.first);
      std::size_t s2 = h(p.second);
      // 结合哈希
      return s1 ^ (s2 + 0x9e3779b97f4a7c15ULL + (s1 << 6) + (s1 >> 2));
    }
  };

  // 将单词按BPE规则切分为token字符串序列（简化实现）
  std::vector<std::string> bpe_tokenize_word(const std::string &word) const;

  // UTF-8 码点切分
  std::vector<std::string> utf8_codepoints(const std::string &text) const;

  // vocab: token -> id
  std::unordered_map<std::string, int> vocab_;
  // reverse vocab: id -> token
  std::unordered_map<int, std::string> inv_vocab_;
  // merges: pair(token_a, token_b) -> rank (越小优先)
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>
      merges_;

  // 空格哨兵 ID（不进入 vocab）
  static constexpr int kSpaceId = -1;

  // 动态 OOV 映射（在 encode 过程中分配新 ID）
  int next_dynamic_id_ = 0;
  std::unordered_map<std::string, int> oov_map_;
  std::unordered_map<int, std::string> inv_oov_map_;
};

} // namespace ow::nn