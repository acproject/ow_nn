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

  // Scratch buffer for temporary allocations inside kernels.
  // Intended for short-lived intermediates (e.g., packed panels).
  void *scratch_alloc(size_t bytes) {
    size_t cur = align_up(scratch_offset, align_bytes);
    if (cur + bytes > scratch.size()) {
      size_t new_size = std::max(scratch.size() * 2, cur + bytes);
      scratch.resize(new_size);
    }
    void *ptr = scratch.data() + cur;
    scratch_offset = cur + bytes;
    return ptr;
  }
  void scratch_reset() { scratch_offset = 0; }
  size_t scratch_used() const { return scratch_offset; }

private:
  std::vector<uint8_t> arena;
  size_t offset = 0;
  size_t align_bytes = 64;
  std::vector<uint8_t> scratch;
  size_t scratch_offset = 0;
};

} // namespace ow::nn