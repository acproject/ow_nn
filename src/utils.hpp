#pragma once

#include "tensor.hpp"
#include "ops.hpp"

namespace ow::nn {
// 参考 Python: x1 = x[..., :L/2]; x2 = x[..., L/2:]; cat((-x2, x1), dim=-1)
static inline TensorPtr rotate_half(const TensorPtr &x) {
  if (!x) throw std::runtime_error("rotate_half: input is null");
  int r = (int)x->shape.size();
  if (r <= 0) throw std::runtime_error("rotate_half: rank must be >= 1");
  int last = r - 1;
  int L = x->shape[last];
  if (L <= 0) throw std::runtime_error("rotate_half: last dim must be > 0");
  if ((L % 2) != 0) throw std::runtime_error("rotate_half: last dim must be even");
  int half = L / 2;

  // x1 = x[..., :half]
  std::vector<int> starts1(r, 0);
  std::vector<int> lens1 = x->shape; lens1[last] = half;
  auto x1 = x->slice_view(starts1, lens1);

  // x2 = x[..., half:]
  std::vector<int> starts2(r, 0); starts2[last] = half;
  std::vector<int> lens2 = x->shape; lens2[last] = L - half;
  auto x2 = x->slice_view(starts2, lens2);

  // -x2
  auto nx2 = x2->elementwise_unary(x2, [](float v){ return -v; });

  // concat([-x2, x1], dim=-1)
  return concat({nx2, x1}, -1);
}
} // namespace ow::nn