#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
namespace ow::nn {
// Align x up to the nearest multiple of a (a must be power of two)
static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

enum class DType {
  FLOAT32,
  INT32,
  FP16,
  BF16,
  INT8,
  Q4_0,
  // 新增更广泛的权重/张量类型支持
  U8,
  BOOL,
  I16,
  I64,
  F64,
  FP8_E4M3,
  FP8_E5M2
};

inline size_t dtype_size(DType t) {
  switch (t) {
  case DType::FLOAT32:
    return 4;
  case DType::INT32:
    return 4;
  case DType::FP16:
    return 2;
  case DType::BF16:
    return 2;
  case DType::INT8:
    return 1;
  case DType::Q4_0:
    return 1;
  case DType::U8:
    return 1;
  case DType::BOOL:
    return 1;
  case DType::I16:
    return 2;
  case DType::I64:
    return 8;
  case DType::F64:
    return 8;
  case DType::FP8_E4M3:
    return 1;
  case DType::FP8_E5M2:
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

// bf16 helpers (use high 16 bits of IEEE754 float32)
static uint16_t float_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

static float bf16_to_float(uint16_t h) {
  uint32_t u = ((uint32_t)h) << 16;
  float out;
  std::memcpy(&out, &u, 4);
  return out;
}
} // namespace ow::nn