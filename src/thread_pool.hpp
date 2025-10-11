#pragma once
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
namespace ow::nn {

class ThreadPool {
public:
  ThreadPool(size_t n = std::thread::hardware_concurrency()) : stop(false) {
    if (n == 0)
      n = 1;
    for (size_t i = 0; i < n; ++i) {
      workers.emplace_back([this] { this->worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lk(mu);
      stop = true;
      cv.notify_all();
    }
    for (auto &t : workers) {
      if (t.joinable())
        t.join();
    }
  }
  template <class F> auto submit(F &&f) -> std::future<decltype(f())> {
    using R = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();
    {
      std::unique_lock<std::mutex> lk(mu);
      tasks.emplace([task] { (*task)(); });
    }
    cv.notify_one();
    return fut;
  }

private:
  bool stop;
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex mu;
  std::condition_variable cv;
  void worker_loop() {
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return stop || !tasks.empty(); });
        if (stop && tasks.empty()) {
          return;
        }
        job = std::move(tasks.front());
        tasks.pop();
      }
      job();
    }
  }
};
} // namespace ow::nn