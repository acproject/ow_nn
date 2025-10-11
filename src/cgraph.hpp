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
#include <vector>
namespace ow::nn {
struct OpNode {
  std::string name;
  std::vector<std::string> inputs;
  std::string output;
  std::function<void(const std::vector<TensorPtr> &,
                     const std::vector<TensorPtr>)>
      fn;
};

struct ComputeGraph {
  std::vector<OpNode> nodes;
  std::unordered_map<std::string, TensorPtr> tensors;
  std::shared_ptr<Context> ctx;
  ComputeGraph(std::shared_ptr<Context> c) : ctx(c) {}
  void add_input(const std::string &name, const TensorPtr &t) {
    tensors[name] = t;
  }

  void add_node(const OpNode &n) { nodes.push_back(n); }
  // simple topo: nodes as given; check dependencies and run ready nodes in
  // parallel
  void run() {
    // build dependency counts
    std::unordered_map<std::string, int> need_count;
    std::unordered_map<std::string, std::vector<int>> consumers;
    for (size_t i = 0; i < nodes.size(); ++i) {
      for (auto &in : nodes[i].inputs) {
        consumers[in].push_back((int)i);
      }
      need_count[nodes[i].name] = (int)nodes[i].inputs.size();
    }
    ThreadPool tp(std::max<size_t>(1, std::thread::hardware_concurrency()));
    std::queue<int> q;
    std::mutex mq;
    std::condition_variable mcv;
    std::atomic<int> remaining((int)nodes.size());
    // initially push nodes with zero inputs
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].inputs.empty())
        q.push((int)i);
    }
    // lambda to schedule node
    auto schedule = [&](int idx) {
      tp.submit(
          [this, idx, &need_count, &consumers, &q, &mq, &mcv, &remaining]() {
            // prepare input tensor pointers
            std::vector<TensorPtr> ins;
            for (auto &iname : nodes[idx].inputs) {
              ins.push_back(this->tensors[iname]);
            }
            // create output tensor placeholder (we allow op to allocate into
            // tensors map)
            // call node function with ins and outs vector (outs empty-> op
            // should create tensor and register)
            nodes[idx].fn(ins, {});
            // after op, assume nodes[idx].output registered in tensors
            // notify consumers
            if (!nodes[idx].output.empty()) {
              auto &outname = nodes[idx].output;
              auto it = consumers.find(outname);
              if (it != consumers.end()) {
                for (int cidx : it->second) {
                  int rem = --need_count[nodes[cidx].name];
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
    while (remaining > 0) {
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
  }
};

} // namespace ow::nn
