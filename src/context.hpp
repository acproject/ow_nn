#pragma once
#include "../include/common.h"
#include <cstddef>
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

class Context : public std::enable_shared_from_this<Context> {
public:
  explicit Context(size_t arena_size = 8 * 1024 * 1024, size_t align = 64)
      : align_bytes(align) {
    arena.resize(arena_size);
    offset = 0;
    // initialize scratch buffer to a smaller default; grows on demand
    // NOTE: legacy shared scratch is kept but unused after TLS migration
    scratch.resize(512 * 1024);
    scratch_offset = 0;
  }

  void *alloc(size_t bytes) {
    size_t cur = align_up(offset, align_bytes);
    if (cur + bytes > arena.size()) {
      // Check if the allocation is too large (>8GB for a single tensor)
      if (bytes > 8ull * 1024ull * 1024ull * 1024ull) {
        std::cerr << "[Context] Error: Single tensor allocation too large (" 
                  << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB). Consider model sharding." << std::endl;
        throw std::bad_alloc();
      }
      
      // Use very conservative growth strategy to avoid system memory exhaustion
      size_t current_size = arena.size();
      size_t needed_size = cur + bytes;
      size_t new_size;
      
      // Check available system memory before attempting allocation
      size_t available_memory = get_available_memory();
      size_t max_safe_allocation = available_memory / 2; // Use at most 50% of available memory
      
      if (current_size > 12ull * 1024ull * 1024ull * 1024ull) { // > 12GB
        // For very large arenas, only grow by exactly what's needed + small buffer
        new_size = needed_size + 256ull * 1024ull * 1024ull; // +256MB buffer
      } else if (current_size > 4ull * 1024ull * 1024ull * 1024ull) { // > 4GB
        // For large arenas, grow conservatively
        new_size = std::min(needed_size + 1024ull * 1024ull * 1024ull, current_size + current_size / 4);
      } else {
        // For smaller arenas, use moderate growth
        new_size = std::max(current_size + current_size / 2, needed_size);
      }
      
      // Ensure we don't exceed safe memory limits
      if (new_size > max_safe_allocation) {
        new_size = std::max(needed_size + 128ull * 1024ull * 1024ull, max_safe_allocation);
        std::cout << "[Context] Limiting allocation to " << (new_size / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB due to system memory constraints" << std::endl;
      }
      
      std::cout << "[Context] Resize arena from " << (current_size / (1024.0 * 1024.0 * 1024.0))
                << " GB to " << (new_size / (1024.0 * 1024.0 * 1024.0)) 
                << " GB (need=" << (needed_size / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
      
      try {
        arena.resize(new_size);
      } catch (const std::bad_alloc& e) {
        std::cerr << "[Context] Failed to resize arena to " << (new_size / (1024.0 * 1024.0 * 1024.0))
                  << " GB" << std::endl;
        
        // Final attempt: allocate exactly what's needed
        size_t minimal_size = needed_size + 64ull * 1024ull * 1024ull; // +64MB buffer
        std::cout << "[Context] Final attempt: minimal resize to " 
                  << (minimal_size / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        
        try {
          arena.resize(minimal_size);
        } catch (const std::bad_alloc& e2) {
          std::cerr << "[Context] Critical: Cannot allocate " 
                    << (minimal_size / (1024.0 * 1024.0 * 1024.0)) 
                    << " GB. Available memory: " << (available_memory / (1024.0 * 1024.0 * 1024.0)) 
                    << " GB" << std::endl;
          throw;
        }
      }
    }
    void *ptr = arena.data() + cur;
    offset = cur + bytes;
    return ptr;
  }

private:
  // Get available system memory in bytes
  size_t get_available_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<size_t>(memInfo.ullAvailPhys);
#else
    // For Linux/Unix systems
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages * page_size);
#endif
  }

  // Thread-local scratch buffer state
  struct ThreadScratch {
    std::vector<uint8_t> buf;
    size_t offset = 0;
  };
  static ThreadScratch& get_tls_scratch() {
    thread_local ThreadScratch ts;
    if (ts.buf.empty()) {
      ts.buf.resize(512 * 1024);
      ts.offset = 0;
    }
    return ts;
  }

public:
  // mark current arena pointer, for simple LIFO lifetime management
  size_t mark() const { return offset; }
  // release arena back to a previous mark (LIFO); does not shrink capacity
  void release_to(size_t m) {
    if (m <= offset) {
      offset = m;
    }
  }
  void reset() { offset = 0; }
  size_t used() const { return offset; }

  size_t capacity() const { return arena.size(); }

  // Enhanced scratch buffer for temporary allocations inside kernels.
  // Thread-safe via thread-local storage. Intended for short-lived intermediates.
  void *scratch_alloc(size_t bytes) {
    auto &ts = get_tls_scratch();
    size_t cur = align_up(ts.offset, align_bytes);
    if (cur + bytes > ts.buf.size()) {
      // Pre-allocate larger chunks for common matrix operations
      size_t min_growth = 16 * 1024 * 1024; // 16MB minimum growth
      size_t needed = cur + bytes;
      size_t new_size;
      
      if (needed < 64 * 1024 * 1024) { // < 64MB
        new_size = std::max(needed * 2, min_growth);
      } else if (needed < 512 * 1024 * 1024) { // < 512MB
        new_size = needed + 128 * 1024 * 1024; // +128MB
      } else {
        new_size = needed + 64 * 1024 * 1024; // +64MB for very large
      }
      
      std::cout << "[Context] Resize scratch buffer from " 
                << (ts.buf.size() / (1024.0 * 1024.0)) << " MB to " 
                << (new_size / (1024.0 * 1024.0)) << " MB" << std::endl;
      ts.buf.resize(new_size);
    }
    void *ptr = ts.buf.data() + cur;
    ts.offset = cur + bytes;
    return ptr;
  }
  
  // Optimized allocation for matrix packing (common use case)
  void *scratch_alloc_matrix_pack(size_t rows, size_t cols, size_t elem_size = sizeof(float)) {
    size_t bytes = rows * cols * elem_size;
    // Align to cache line boundaries for better performance
    size_t aligned_bytes = align_up(bytes, 64);
    return scratch_alloc(aligned_bytes);
  }
  
  void scratch_reset() { auto &ts = get_tls_scratch(); ts.offset = 0; }
  size_t scratch_used() const { return get_tls_scratch().offset; }
  size_t scratch_capacity() const { return get_tls_scratch().buf.size(); }

private:
  std::vector<uint8_t> arena;
  size_t offset = 0;
  size_t align_bytes = 64;
  // legacy shared scratch (unused after TLS migration, kept to avoid breaking ABI)
  std::vector<uint8_t> scratch;
  size_t scratch_offset = 0;
};

} // namespace ow::nn