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
  void reset() { offset = 0; }
  size_t used() const { return offset; }

  size_t capacity() const { return arena.size(); }

private:
  std::vector<uint8_t> arena;
  size_t offset = 0;
  size_t align_bytes = 64;
};

} // namespace ow::nn