#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include "src/tensor.hpp"
#include "src/context.hpp"

using namespace ow::nn;

// 性能测试辅助函数
class BenchmarkTimer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // 返回毫秒
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// 创建随机测试矩阵
TensorPtr create_random_tensor(const std::vector<int>& shape, std::shared_ptr<Context> ctx) {
    auto tensor = Tensor::create(ctx, shape, DType::FLOAT32);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float* data = static_cast<float*>(tensor->data);
    for (size_t i = 0; i < tensor->nelements(); ++i) {
        data[i] = dis(gen);
    }
    
    return tensor;
}

// 矩阵乘法性能测试
void benchmark_matmul(const std::string& test_name, int M, int K, int N, int iterations = 10) {
    std::cout << "\n=== " << test_name << " ===\n";
    std::cout << "Matrix sizes: A[" << M << "x" << K << "] × B[" << K << "x" << N << "]\n";
    
    auto ctx = std::make_shared<Context>();
    
    // 创建测试矩阵
    auto A = create_random_tensor({M, K}, ctx);
    auto B = create_random_tensor({K, N}, ctx);
    
    BenchmarkTimer timer;
    
    // 测试原始的 matmul_blocked_mt
    std::vector<double> blocked_times;
    for (int i = 0; i < iterations; ++i) {
        ctx->scratch_reset(); // 重置scratch buffer
        timer.start();
        auto result1 = Tensor::matmul_blocked_mt(A, B);
        double time = timer.stop();
        blocked_times.push_back(time);
    }
    
    // 测试优化的 matmul_cache_friendly
    std::vector<double> cache_friendly_times;
    for (int i = 0; i < iterations; ++i) {
        ctx->scratch_reset(); // 重置scratch buffer
        timer.start();
        auto result2 = Tensor::matmul_cache_friendly(A, B);
        double time = timer.stop();
        cache_friendly_times.push_back(time);
    }
    
    // 计算统计数据
    double blocked_sum = 0, blocked_min = blocked_times[0], blocked_max = blocked_times[0];
    for (size_t i = 0; i < blocked_times.size(); ++i) {
        double t = blocked_times[i];
        blocked_sum += t;
        if (t < blocked_min) blocked_min = t;
        if (t > blocked_max) blocked_max = t;
    }
    double blocked_avg = blocked_sum / blocked_times.size();
    
    double cache_sum = 0, cache_min = cache_friendly_times[0], cache_max = cache_friendly_times[0];
    for (size_t i = 0; i < cache_friendly_times.size(); ++i) {
        double t = cache_friendly_times[i];
        cache_sum += t;
        if (t < cache_min) cache_min = t;
        if (t > cache_max) cache_max = t;
    }
    double cache_avg = cache_sum / cache_friendly_times.size();
    
    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "matmul_blocked_mt:     " << blocked_avg << "ms (min: " << blocked_min << ", max: " << blocked_max << ")\n";
    std::cout << "matmul_cache_friendly: " << cache_avg << "ms (min: " << cache_min << ", max: " << cache_max << ")\n";
    
    double speedup = blocked_avg / cache_avg;
    std::cout << "Speedup: " << speedup << "x ";
    if (speedup > 1.0) {
        std::cout << "(+" << ((speedup - 1.0) * 100) << "% faster)\n";
    } else {
        std::cout << "(" << ((1.0 - speedup) * 100) << "% slower)\n";
    }
    
    // 计算GFLOPS
    double gflops_ops = 2.0 * M * K * N / 1e9; // 矩阵乘法的浮点运算数
    std::cout << "GFLOPS (cache_friendly): " << (gflops_ops / (cache_avg / 1000.0)) << "\n";
}

int main() {
    std::cout << "=== 矩阵乘法性能基准测试 ===\n";
    std::cout << "对比 matmul_blocked_mt vs matmul_cache_friendly\n";
    
    // 先测试小矩阵
    benchmark_matmul("Small matrices (典型attention)", 64, 256, 256, 3);
    benchmark_matmul("Medium matrices", 32, 512, 512, 3);
    benchmark_matmul("Large matrices (MLP)", 16, 1024, 1024, 3);
    benchmark_matmul("Transformer attention", 128, 512, 512, 3);
    benchmark_matmul("Batch processing", 256, 256, 256, 3);
    benchmark_matmul("Wide matrices", 64, 128, 1024, 3);
    
    std::cout << "\n=== 测试完成 ===\n";
    return 0;
}