#pragma once
/// ------------------- compute graph ----------------------
#include "context.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace ow::nn {
struct OpNode {
  std::string name;
  std::vector<std::string> inputs;
  std::string output;
  std::unordered_map<std::string, std::string> attrs;
  // 运行函数：根据输入返回输出张量（框架负责注册到图）
  std::function<TensorPtr(const std::vector<TensorPtr> &,
                          const std::unordered_map<std::string, std::string> &)> fn;
  // 形状/类型推断（可选）：根据输入描述与属性推断输出描述
  std::function<TensorDesc(const std::vector<TensorDesc> &,
                           const std::unordered_map<std::string, std::string> &)> infer;
};

struct ComputeGraph {
  std::vector<OpNode> nodes;
  std::unordered_map<std::string, TensorPtr> tensors;
  std::unordered_map<std::string, TensorDesc> descs;
  std::shared_ptr<Context> ctx;
  ComputeGraph(std::shared_ptr<Context> c) : ctx(c) {}
  void add_input(const std::string &name, const TensorPtr &t) {
    tensors[name] = t;
    descs[name] = t->desc();
  }

  void add_node(const OpNode &n) { nodes.push_back(n); }
  
  TensorPtr get_tensor(const std::string &name) {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
      return it->second;
    }
    return nullptr;
  }
  // simple topo: nodes as given; check dependencies and run ready nodes in
  // parallel
  void run() {
    // build dependency counts
    std::vector<int> need_count(nodes.size(), 0);
    std::unordered_map<std::string, std::vector<int>> consumers;
    for (size_t i = 0; i < nodes.size(); ++i) {
      for (auto &in : nodes[i].inputs) {
        consumers[in].push_back((int)i);
      }
      // 未满足的依赖：仅统计那些在 tensors 中尚不存在的输入
      int unmet = 0;
      for (auto &in : nodes[i].inputs) {
        if (tensors.find(in) == tensors.end()) {
          ++unmet;
        }
      }
      need_count[i] = unmet;
    }
    ThreadPool tp(std::max<size_t>(1, std::thread::hardware_concurrency()));
    std::queue<int> q;
    std::mutex mq;
    std::condition_variable mcv;
    std::atomic<int> remaining((int)nodes.size());
    std::atomic<bool> failed(false);
    std::string err_msg;
    std::mutex err_m;
    // initially push nodes with all deps satisfied (包括已有输入)
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (need_count[i] == 0)
        q.push((int)i);
    }
    // lambda to schedule node
    auto schedule = [&](int idx) {
      tp.submit(
          [this, idx, &need_count, &consumers, &q, &mq, &mcv, &remaining, &failed, &err_msg, &err_m]() {
            if (failed) return;
            // prepare input tensor pointers
            std::vector<TensorPtr> ins;
            for (auto &iname : nodes[idx].inputs) {
              ins.push_back(this->tensors[iname]);
            }
            try {
              // Run op and register output
              TensorPtr out = nodes[idx].fn(ins, nodes[idx].attrs);
              if (!out) throw std::runtime_error("node returned null output");
              this->tensors[nodes[idx].output] = out;
              // infer desc if available, otherwise from tensor
              std::vector<TensorDesc> in_descs;
              in_descs.reserve(ins.size());
              for (auto &t : ins) in_descs.push_back(t->desc());
              if (nodes[idx].infer) {
                this->descs[nodes[idx].output] = nodes[idx].infer(in_descs, nodes[idx].attrs);
              } else {
                this->descs[nodes[idx].output] = out->desc();
              }
            } catch (const std::exception &ex) {
              std::lock_guard<std::mutex> lk(err_m);
              failed = true;
              err_msg = std::string("[Graph Error] node '") + nodes[idx].name + "': " + ex.what();
            }
            // notify consumers
            if (!nodes[idx].output.empty()) {
              auto &outname = nodes[idx].output;
              auto it = consumers.find(outname);
              if (it != consumers.end()) {
                for (int cidx : it->second) {
                  int rem = --need_count[cidx];
                  if (rem == 0) {
                    // push to queue
                    std::unique_lock<std::mutex> lk(mq);
                    q.push(cidx);
                    mcv.notify_one();
                  }
                }
              }
            }
            --remaining;
          });
    };
    // Schedule initial ready nodes
    while (!q.empty()) {
      int i = q.front();
      q.pop();
      schedule(i);
    }
    // Wait for remaining tasks to finish; newly ready nodes get pushed by
    // consumers and will be scheduled by waiting thread below.
    while (remaining > 0 && !failed) {
      int idx = -1;
      {
        std::unique_lock<std::mutex> lk(mq);
        if (q.empty())
          mcv.wait_for(lk, std::chrono::milliseconds(1));
        if (!q.empty()) {
          idx = q.front();
          q.pop();
        }
      }
      if (idx >= 0)
        schedule(idx);
    }
    if (failed) {
      throw std::runtime_error(err_msg);
    }
  }

  // 验证图结构：检查上下文、输出重名、缺失输入以及潜在环
  void validate() {
    if (!ctx) throw std::runtime_error("ComputeGraph: ctx is null");
    // 输出名必须非空且唯一；节点必须具备可运行函数
    std::unordered_set<std::string> outputs;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].output.empty())
        throw std::runtime_error("validate: node output name empty");
      if (outputs.count(nodes[i].output))
        throw std::runtime_error(std::string("validate: duplicate output '") + nodes[i].output + "'");
      outputs.insert(nodes[i].output);
      if (!nodes[i].fn)
        throw std::runtime_error(std::string("validate: node '") + nodes[i].name + "' has no fn");
    }
    // Kahn 算法：以现有输入为可用集合，逐步解析依赖，若无法遍历完所有节点则存在环或缺失输入
    std::unordered_set<std::string> available;
    for (auto &kv : tensors) available.insert(kv.first);
    std::unordered_map<std::string, std::vector<int>> consumers;
    std::vector<int> indeg(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); ++i) {
      int unmet = 0;
      for (auto &in : nodes[i].inputs) {
        consumers[in].push_back((int)i);
        if (!available.count(in)) ++unmet;
      }
      indeg[i] = unmet;
    }
    std::queue<int> q;
    for (size_t i = 0; i < nodes.size(); ++i)
      if (indeg[i] == 0) q.push((int)i);
    int visited = 0;
    while (!q.empty()) {
      int u = q.front(); q.pop();
      ++visited;
      available.insert(nodes[u].output);
      auto it = consumers.find(nodes[u].output);
      if (it != consumers.end()) {
        for (int v : it->second) {
          int rem = --indeg[v];
          if (rem == 0) q.push(v);
        }
      }
    }
    if (visited != (int)nodes.size()) {
      throw std::runtime_error("validate: graph has cycles or missing inputs; unresolved nodes count=" + std::to_string(nodes.size() - visited));
    }
  }
};

} // namespace ow::nn
