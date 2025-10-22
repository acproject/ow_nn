#pragma once
#include "context.hpp"
#include "lazy_safetensors_loader.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ow::nn {

class MemoryOptimizer {
public:
    // Get available system memory in GB
    static double get_available_memory_gb() {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return memInfo.ullAvailPhys / (1024.0 * 1024.0 * 1024.0);
#else
        long pages = sysconf(_SC_AVPHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return (pages * page_size) / (1024.0 * 1024.0 * 1024.0);
#endif
    }
    
    // Print current memory status
    static void print_memory_status(const std::string& prefix = "") {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        double total_gb = memInfo.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
        double avail_gb = memInfo.ullAvailPhys / (1024.0 * 1024.0 * 1024.0);
        double used_gb = total_gb - avail_gb;
        std::cout << "[Memory]" << (prefix.empty() ? "" : " " + prefix + ":") 
                  << " Total: " << total_gb << " GB, Used: " << used_gb 
                  << " GB, Available: " << avail_gb << " GB" << std::endl;
#endif
    }
    
    // Determine optimal initial context size based on available memory
    static size_t get_optimal_initial_context_size() {
        double available_gb = get_available_memory_gb();
        
        // More aggressive initial sizing for large models to reduce reallocations
        if (available_gb >= 48.0) {
            return 16ULL * 1024 * 1024 * 1024; // 16GB for very high-memory systems
        } else if (available_gb >= 32.0) {
            return 12ULL * 1024 * 1024 * 1024; // 12GB for high-memory systems
        } else if (available_gb >= 24.0) {
            return 8ULL * 1024 * 1024 * 1024;  // 8GB for medium-high memory systems
        } else if (available_gb >= 16.0) {
            return 6ULL * 1024 * 1024 * 1024;  // 6GB for medium-memory systems
        } else if (available_gb >= 8.0) {
            return 3ULL * 1024 * 1024 * 1024;  // 3GB for low-memory systems
        } else {
            return 1ULL * 1024 * 1024 * 1024;  // 1GB minimum
        }
    }
    
    // Load weights in batches to manage memory usage
    static std::unordered_map<std::string, TensorPtr> load_weights_optimized(
        LazySafetensorsLoader& loader,
        std::shared_ptr<Context> ctx,
        const std::vector<std::string>& weight_patterns,
        bool use_memory_mapping = true) {
        
        std::unordered_map<std::string, TensorPtr> weights;
        double initial_memory = get_available_memory_gb();
        
        std::cout << "[MemOpt] Starting optimized weight loading..." << std::endl;
        print_memory_status("Initial");
        
        // Categorize weights by size and importance
        std::vector<std::string> essential_weights;
        std::vector<std::string> layer_weights;
        
        for (const auto& name : loader.names()) {
            bool matches_pattern = false;
            for (const auto& pattern : weight_patterns) {
                if (name.find(pattern) != std::string::npos) {
                    matches_pattern = true;
                    break;
                }
            }
            
            if (!matches_pattern) continue;
            
            // Categorize weights
            if (name.find("embed_tokens") != std::string::npos ||
                name.find("wte") != std::string::npos ||
                name.find("lm_head") != std::string::npos ||
                name.find("model.norm") != std::string::npos ||
                name.find("norm.weight") != std::string::npos) {
                essential_weights.push_back(name);
            } else if (name.find("language_model.layers") != std::string::npos ||
                       name.find("model.layers") != std::string::npos ||
                       name.find("layers.") != std::string::npos) {
                layer_weights.push_back(name);
            }
        }
        
        // Load essential weights first
        std::cout << "[MemOpt] Loading " << essential_weights.size() << " essential weights..." << std::endl;
        for (const auto& name : essential_weights) {
            try {
                auto tensor = loader.make_tensor(name, ctx, !use_memory_mapping);
                if (tensor) {
                    weights[name] = tensor;
                    std::cout << "[MemOpt][OK] " << name << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[MemOpt][SKIP] " << name << " - " << e.what() << std::endl;
            }
        }
        
        print_memory_status("After essential weights");
        
        // Load layer weights in batches
        const size_t batch_size = 10; // Load 10 layers at a time
        std::cout << "[MemOpt] Loading " << layer_weights.size() << " layer weights in batches..." << std::endl;
        
        for (size_t i = 0; i < layer_weights.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, layer_weights.size());
            std::cout << "[MemOpt] Loading batch " << (i / batch_size + 1) 
                      << " (weights " << i << "-" << (end - 1) << ")" << std::endl;
            
            for (size_t j = i; j < end; ++j) {
                const auto& name = layer_weights[j];
                try {
                    auto tensor = loader.make_tensor(name, ctx, !use_memory_mapping);
                    if (tensor) {
                        weights[name] = tensor;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[MemOpt][SKIP] " << name << " - " << e.what() << std::endl;
                }
            }
            
            // Check memory usage after each batch
            double current_memory = get_available_memory_gb();
            double memory_used = initial_memory - current_memory;
            
            if (memory_used > initial_memory * 0.7) { // Used more than 70% of initial memory
                std::cout << "[MemOpt] Warning: High memory usage detected (" 
                          << memory_used << " GB used)" << std::endl;
                print_memory_status("Batch " + std::to_string(i / batch_size + 1));
            }
        }
        
        print_memory_status("Final");
        std::cout << "[MemOpt] Loaded " << weights.size() << " weights successfully" << std::endl;
        
        return weights;
    }
};

} // namespace ow::nn