#pragma once
#include "../include/common.h"
#include <cstddef>
#include <memory>
#include <iostream>

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
      // Check if the allocation is too large (>10GB for a single tensor)
      if (bytes > 10ull * 1024ull * 1024ull * 1024ull) {
        std::cerr << "[Context] Warning: Attempting to allocate very large tensor (" 
                  << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
        throw std::bad_alloc();
      }
      
      // Use more conservative growth strategy for large arenas
      size_t current_size = arena.size();
      size_t needed_size = cur + bytes;
      size_t new_size;
      
      if (current_size > 16ull * 1024ull * 1024ull * 1024ull) { // > 16GB
        // For very large arenas, only grow by what's needed + 1GB buffer
        new_size = needed_size + 1024ull * 1024ull * 1024ull;
      } else if (current_size > 8ull * 1024ull * 1024ull * 1024ull) { // > 8GB
        // For large arenas, grow by 1.5x or needed size, whichever is smaller
        new_size = std::min(current_size + current_size / 2, needed_size + 2ull * 1024ull * 1024ull * 1024ull);
      } else {
        // For smaller arenas, use the original doubling strategy
        new_size = std::max(current_size * 2, needed_size);
      }
      
      std::cout << "[Context] Resize arena from " << current_size << " to " << new_size
                << " (need=" << needed_size << ")" << std::endl;
      
      try {
        arena.resize(new_size);
      } catch (const std::bad_alloc& e) {
        std::cerr << "[Context] Failed to resize arena to " << new_size 
                  << " bytes (" << (new_size / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
        
        // Try a more conservative approach: just allocate what's needed + small buffer
        size_t minimal_size = needed_size + 512ull * 1024ull * 1024ull; // +512MB buffer
        std::cout << "[Context] Trying minimal resize to " << minimal_size 
                  << " bytes (" << (minimal_size / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
        
        try {
          arena.resize(minimal_size);
        } catch (const std::bad_alloc& e2) {
          std::cerr << "[Context] Failed minimal resize, allocation impossible" << std::endl;
          throw;
        }
      }
    }
    void *ptr = arena.data() + cur;
    offset = cur + bytes;
    return ptr;
  }
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