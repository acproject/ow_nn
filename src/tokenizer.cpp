#include "../include/tokenizer.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#elif __has_include("../build/_deps/json-src/single_include/nlohmann/json.hpp")
#include "../build/_deps/json-src/single_include/nlohmann/json.hpp"
#elif __has_include("../build/_deps/json-src/include/nlohmann/json.hpp")
#include "../build/_deps/json-src/include/nlohmann/json.hpp"
#else
#error "nlohmann/json.hpp not found; ensure dependency is available"
#endif
#include <string>
#include <vector>
using json = nlohmann::json;

namespace ow::nn {
static inline std::string read_file(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("cannot open file: " + path);
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}
Tokenizer::Tokenizer(const std::string &vocab_path,
                     const std::string &merges_path) {

  // 载入 vocab.json
  std::string vocab_text = read_file(vocab_path);
  json j = json::parse(vocab_text);
  for (auto it = j.begin(); it != j.end(); ++it) {
    std::string token = it.key();
    int id = it.value().get<int>();
    // 将 token 字符串映射到对应的 id，存入 vocab_ 字典
    vocab_.emplace(token, id);
    inv_vocab_.emplace(id, token);
  }
  // 初始化动态 OOV 起始 ID：
  // 选择当前 vocab 最大 ID + 1 的值作为起点
  int max_id = -1;
  for (const auto &kv : vocab_) max_id = std::max(max_id, kv.second);
  next_dynamic_id_ = max_id + 1;

  // 载入 merges.txt
  std::string merges_text = read_file(merges_path);
  // 将 merges.txt 内容读入字符串流以便逐行解析
  std::istringstream ms(merges_text);
  std::string line;
  int rank = 0;
  while (std::getline(ms, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ls(line);
    std::string a, b;
    if (!(ls >> a >> b))
      continue;
    merges_[{a, b}] = rank++;
  }
}

std::vector<std::string> Tokenizer::bpe_tokenize_word(const std::string &word) const {
  // 初始拆分为单字符 token
  std::vector<std::string> tokens;
  for (size_t i = 0; i < word.size(); ++i) {
    tokens.push_back(std::string(1, word[i]));
  }

  // 循环合并：不断寻找当前 tokens 中“相邻二元组”在 merges_
  // 字典里优先级最高（rank 最小）的一对， 将其合并为一个新
  // token，直到找不到可合并的相邻对为止。
  while (true) {
    int best_rank = INT32_MAX; // 记录当前最小 rank
    int best_pos = -1;         // 记录该最小 rank 对应的起始下标
    // 遍历所有相邻二元组
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      auto it = merges_.find({tokens[i], tokens[i + 1]});
      if (it != merges_.end() && it->second < best_rank) {
        best_rank = it->second;         // 更新最小 rank
        best_pos = static_cast<int>(i); // 记录位置
      }
    }
    // 若找不到可合并的相邻对，结束循环
    if (best_pos == -1)
      break;
    // 合并：把 best_pos 与 best_pos+1 两个 token 拼接，并删除后者
    tokens[best_pos] = tokens[best_pos] + tokens[best_pos + 1];
    tokens.erase(tokens.begin() + best_pos + 1);
  }
  return tokens;
}

// UTF-8 码点切分
std::vector<std::string> Tokenizer::utf8_codepoints(const std::string &text) const {
  std::vector<std::string> cps;
  const unsigned char *s = reinterpret_cast<const unsigned char *>(text.data());
  size_t i = 0, n = text.size();
  while (i < n) {
    unsigned char c = s[i];
    size_t len = 1;
    if ((c & 0x80) == 0x00) {
      len = 1;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < n) {
      len = 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < n) {
      len = 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < n) {
      len = 4;
    }
    cps.emplace_back(text.substr(i, len));
    i += len;
  }
  return cps;
}

// encode 函数：把输入文本 text 转换成对应的 token id 序列
// 1. 用 istringstream 按空格切分出一个个“词”
// 2. 对每个词调用 bpe_tokenize_word，得到 BPE 子词列表
// 3. 将每个子词查 vocab_ 得到 id；若找不到，则回退到 <|unknown|> 的 id
// 4. 最终返回整段文本对应的 id 序列
std::vector<int> Tokenizer::encode(const std::string &text) {
  std::vector<int> ids;
  // 按空白分段但保留空格为哨兵
  std::string current;
  auto flush_word = [&](const std::string &w) {
    if (w.empty()) return;
    // BPE 子词
    auto tokens = bpe_tokenize_word(w);
    for (auto &t : tokens) {
      auto it = vocab_.find(t);
      if (it != vocab_.end()) {
        ids.push_back(it->second);
      } else {
        // 未命中：为 OOV 分配动态 ID
        auto oit = oov_map_.find(t);
        if (oit == oov_map_.end()) {
          int nid = next_dynamic_id_++;
          oov_map_[t] = nid;
          inv_oov_map_[nid] = t;
          ids.push_back(nid);
        } else {
          ids.push_back(oit->second);
        }
      }
    }
  };

  for (size_t i = 0; i < text.size(); ++i) {
    char ch = text[i];
    if (ch == ' ') {
      flush_word(current);
      current.clear();
      ids.push_back(kSpaceId);
    } else {
      current.push_back(ch);
    }
  }
  flush_word(current);
  return ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const {
  std::ostringstream oss;
  for (int id : ids) {
    if (id == kSpaceId) {
      oss << ' ';
      continue;
    }
    if (auto it = inv_vocab_.find(id); it != inv_vocab_.end()) {
      oss << it->second;
      continue;
    }
    if (auto oit = inv_oov_map_.find(id); oit != inv_oov_map_.end()) {
      oss << oit->second;
    }
  }
  return oss.str();
}

} // namespace ow::nn