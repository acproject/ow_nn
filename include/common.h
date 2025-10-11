#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
namespace ow::nn {
// 将输入值 x 向上对齐到 a 的整数倍
// 参数:
//   x - 需要对齐的原始值
//   a - 对齐单位，必须是 2 的幂
// 返回:
//   大于等于 x 且为 a 的整数倍的最小值
static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

enum class DType {
  FLOAT32,
  INT32,
  FP16,
  INT8,
  Q4_0
};

inline size_t dtype_size(DType t) {
  switch (t) {
  case DType::FLOAT32:
    return 4;
  case DType::INT32:
    return 4;
  case DType::FP16:
    return 2;
  case DType::INT8:
    return 1;
  case DType::Q4_0:
    return 1;

  }
  return 1;
}

// fp16 helpers
static uint16_t float_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 31) & 0x1;
  int32_t exp = ((x >> 23) & 0xFF) - 127;
  uint32_t mant = x & 0x7FFFFF;
  uint16_t h;
  if (exp < -24) {
    h = (uint16_t)(sign << 15);
  } else if (exp < -14) {
    mant |= 0x800000;
    int shift = -14 - exp;
    uint16_t m = (uint16_t)(mant >> (13 + shift));
    h = (uint16_t) ((sign << 15)|m);
  } else if(exp > 15) {
    h = (uint16_t) ((sign << 15) | (0x1F << 10));
  } else {
    uint16_t e = (uint16_t)(exp + 15);
    uint16_t m = (uint16_t)(mant >> 13);
    h = (uint16_t)((sign << 15) | (e << 10) | m);
  }
  return h;
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            int e = -14; 
            uint32_t m = mant;
            while((m & 0x400) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FF;
            uint32_t exp_f = (e + 127) & 0xFF;
            uint32_t mant_f = m << 13;
            f = (sign << 31) | (exp_f << 23) | mant_f;
        }
    } else if (exp == 0x1F) {
        f = (sign <<31) | (0xFF <<23) | (mant << 13);
    } else {
        uint32_t exp_f = (exp - 15 +127) & 0xFF;
        uint32_t mant_f = mant << 13;
        f = (sign<<31) | (exp_f << 23) | mant_f;
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
} // namespace ow::nn