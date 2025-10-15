#pragma once
#include "../include/common.h"
#include <cstddef>
#include <memory>

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
      size_t new_size = std::max(arena.size() * 2, cur + bytes);
      arena.resize(new_size);
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