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
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
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
    if (bytes == 0) return nullptr;
    
    // Check for extremely large single tensor allocations (>8GB)
    if (bytes > 8ULL * 1024 * 1024 * 1024) {
      std::cerr << "[Context] Error: Single tensor allocation too large ("
                << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
      throw std::bad_alloc();
    }
    
    // Align to 64-byte boundary for better performance
    size_t aligned_bytes = (bytes + 63) & ~63ULL;
    
    // Check if we need to resize
    if (offset + aligned_bytes > arena.size()) {
      size_t needed_size = offset + aligned_bytes;
      size_t current_size = arena.size();
      
      // More aggressive growth strategy to reduce frequent reallocations
      // For large models, grow by at least 2GB or 100% of current size, whichever is larger
      size_t min_growth_gb = 2ULL * 1024 * 1024 * 1024; // 2GB minimum growth
      size_t percentage_growth = current_size; // 100% growth
      size_t growth = std::max({min_growth_gb, percentage_growth, needed_size - current_size});
      size_t new_size = current_size + growth;
      
      // Check available system memory and limit allocation if necessary
      size_t available_memory = get_available_memory();
      size_t max_safe_size = static_cast<size_t>(available_memory * 0.75); // Use 75% of available memory
      
      if (new_size > max_safe_size) {
        // If we can't grow aggressively, try a more conservative approach
        size_t conservative_growth = std::max(needed_size - current_size, current_size / 4);
        new_size = std::min(current_size + conservative_growth, max_safe_size);
        
        if (new_size < needed_size) {
          new_size = needed_size; // Must allocate at least what's needed
        }
        
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
                  << " GB. Trying minimal resize..." << std::endl;
        
        // Fallback: try to allocate just what we need
        try {
          arena.resize(needed_size);
          std::cout << "[Context] Minimal resize successful: " << (needed_size / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        } catch (const std::bad_alloc& e2) {
          std::cerr << "[Context] Critical: Cannot allocate even minimal memory (" 
                    << (needed_size / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
          throw;
        }
      }
    }
    
    void* ptr = arena.data() + offset;
    offset += aligned_bytes;
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
#elif defined(__APPLE__)
    // macOS: use mach host_statistics64 to get free and inactive pages
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stats;
    kern_return_t kr = host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                         reinterpret_cast<host_info64_t>(&vm_stats),
                                         &count);
    if (kr != KERN_SUCCESS) {
      return 0;
    }
    // Get page size via sysctl, fallback to sysconf if needed
    int page_size = 0;
    size_t len = sizeof(page_size);
    int mib[2] = {CTL_HW, HW_PAGESIZE};
    if (sysctl(mib, 2, &page_size, &len, nullptr, 0) != 0 || page_size <= 0) {
      page_size = (int)sysconf(_SC_PAGE_SIZE);
    }
    uint64_t free_pages = vm_stats.free_count + vm_stats.inactive_count;
    return static_cast<size_t>(free_pages) * static_cast<size_t>(page_size);
#else
    // Linux/Unix (non-Apple): use available physical pages
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
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