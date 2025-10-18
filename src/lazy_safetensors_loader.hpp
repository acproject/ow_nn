#pragma once
#include "../include/common.h"
#include "context.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include <cmath>
#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#elif __has_include("../build/_deps/json-src/single_include/nlohmann/json.hpp")
#include "../build/_deps/json-src/single_include/nlohmann/json.hpp"
#elif __has_include("../build/_deps/json-src/include/nlohmann/json.hpp")
#include "../build/_deps/json-src/include/nlohmann/json.hpp"
#else
#error "nlohmann/json.hpp not found; ensure dependency is available"
#endif
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace ow::nn {
using json = nlohmann::json;

// Lazy memory-mapped file that only maps when needed
#ifdef _WIN32
struct LazyMMapFile {
  std::wstring wpath;
  mutable HANDLE hFile = nullptr;
  mutable HANDLE hMap = nullptr;
  mutable void *base = nullptr;
  size_t size = 0;
  mutable bool mapped = false;
  
  static std::wstring to_w(const std::string &u8) {
    int n = MultiByteToWideChar(CP_UTF8, 0, u8.c_str(), (int)u8.size(), NULL, 0);
    std::wstring w;
    w.resize(n);
    MultiByteToWideChar(CP_UTF8, 0, u8.c_str(), (int)u8.size(), w.data(), n);
    return w;
  }
  
  explicit LazyMMapFile(const std::string &path) {
    wpath = to_w(path);
    // Only open file to get size, don't map yet
    HANDLE tempFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (tempFile == INVALID_HANDLE_VALUE)
      throw std::runtime_error("open fail:" + path);
    
    LARGE_INTEGER li;
    if (!GetFileSizeEx(tempFile, &li)) {
      CloseHandle(tempFile);
      throw std::runtime_error("size fail:" + path);
    }
    size = (size_t)li.QuadPart;
    CloseHandle(tempFile);
  }
  
  void* get_data() const {
    if (!mapped) {
      map_file();
    }
    return base;
  }
  
  void map_file() const {
    if (mapped) return;
    
    hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
      throw std::runtime_error("lazy open fail");
    
    hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
      CloseHandle(hFile);
      throw std::runtime_error("lazy map fail");
    }
    
    base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
      CloseHandle(hMap);
      CloseHandle(hFile);
      throw std::runtime_error("lazy map view fail");
    }
    mapped = true;
  }
  
  void unmap() const {
    if (!mapped) return;
    
    if (base) {
      UnmapViewOfFile(base);
      base = nullptr;
    }
    if (hMap) {
      CloseHandle(hMap);
      hMap = nullptr;
    }
    if (hFile && hFile != INVALID_HANDLE_VALUE) {
      CloseHandle(hFile);
      hFile = nullptr;
    }
    mapped = false;
  }
  
  ~LazyMMapFile() {
    unmap();
  }
};
#else
struct LazyMMapFile {
  std::string path;
  mutable std::vector<uint8_t> buf;
  mutable void *base = nullptr;
  size_t size = 0;
  mutable bool loaded = false;
  
  explicit LazyMMapFile(const std::string &p) : path(p) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("open fail:" + path);
    size = ifs.tellg();
  }
  
  void* get_data() const {
    if (!loaded) {
      std::ifstream ifs(path, std::ios::binary);
      if (!ifs) throw std::runtime_error("lazy open fail:" + path);
      buf.resize(size);
      ifs.read(reinterpret_cast<char*>(buf.data()), size);
      base = buf.data();
      loaded = true;
    }
    return base;
  }
  
  void unmap() const {
    if (loaded) {
      buf.clear();
      buf.shrink_to_fit();
      base = nullptr;
      loaded = false;
    }
  }
};
#endif

struct LazySTEntry {
  std::string st_dtype;
  std::vector<int> shape;
  size_t nbytes;
  size_t file_offset;
  std::string file_path;  // Store file path for lazy loading
};

struct LazySTFile {
  std::string path;
  std::shared_ptr<LazyMMapFile> mm;
  uint64_t header_len = 0;
  size_t data_base = 0;
  std::unordered_map<std::string, LazySTEntry> entries;
  mutable bool header_parsed = false;
  
  void parse_header() const {
    if (header_parsed) return;
    
    // Only map enough to read header
    mm->map_file();
    auto* data = static_cast<const uint8_t*>(mm->get_data());
    
    if (mm->size < 8) throw std::runtime_error("file too small");
    
    const_cast<LazySTFile*>(this)->header_len = *reinterpret_cast<const uint64_t*>(data);
    if (header_len > mm->size - 8) throw std::runtime_error("invalid header len");
    
    std::string header_str(reinterpret_cast<const char*>(data + 8), header_len);
    json j = json::parse(header_str);
    
    const_cast<LazySTFile*>(this)->data_base = 8 + header_len;
    
    for (auto &kv : j.items()) {
      if (kv.key() == "__metadata__") continue;
      
      LazySTEntry entry;
      entry.st_dtype = kv.value()["dtype"].get<std::string>();
      entry.shape = kv.value()["shape"].get<std::vector<int>>();
      entry.file_offset = kv.value()["data_offsets"][0].get<size_t>();
      entry.nbytes = kv.value()["data_offsets"][1].get<size_t>() - entry.file_offset;
      entry.file_path = path;
      
      const_cast<std::unordered_map<std::string, LazySTEntry>&>(entries)[kv.key()] = entry;
    }
    
    // Unmap after reading header to save memory
    mm->unmap();
    header_parsed = true;
  }
};

struct LazySafetensorsLoader {
  std::vector<LazySTFile> files;
  std::unordered_map<std::string, std::pair<int, LazySTEntry>> index;
  
  void load_dir(const std::string &dir) {
    files.clear();
    index.clear();
    
    // Check for index file first
    std::filesystem::path idx_path;
    for (auto &de : std::filesystem::directory_iterator(dir)) {
      if (!de.is_regular_file()) continue;
      auto p = de.path();
      auto fname = p.filename().string();
      if (fname.size() >= 26 && fname.find(".safetensors.index.json") != std::string::npos) {
        idx_path = p;
        break;
      }
    }
    
    if (!idx_path.empty()) {
      std::ifstream ifs(idx_path.string());
      if (!ifs) throw std::runtime_error("open index fail:" + idx_path.string());
      json j = json::parse(ifs);
      
      if (j.contains("weight_map")) {
        std::unordered_set<std::string> shard_files;
        for (auto &kv : j["weight_map"].items()) {
          std::string shard = kv.value().get<std::string>();
          shard_files.insert(shard);
        }
        
        // Check if files exist but don't load them yet
        std::vector<std::string> missing;
        for (const auto &s : shard_files) {
          auto sp = (std::filesystem::path(dir) / s).string();
          if (!std::filesystem::exists(sp)) {
            missing.push_back(sp);
          }
        }
        
        if (!missing.empty()) {
          std::string msg = "safetensors shards missing (" + std::to_string(missing.size()) + "): ";
          for (size_t i = 0; i < std::min<size_t>(missing.size(), 3); i++) {
            msg += missing[i] + "; ";
          }
          throw std::runtime_error(msg);
        }
        
        // Create lazy file objects
        for (const auto &s : shard_files) {
          auto sp = (std::filesystem::path(dir) / s).string();
          LazySTFile f;
          f.path = sp;
          f.mm = std::make_shared<LazyMMapFile>(f.path);
          files.emplace_back(std::move(f));
        }
        
        // Parse headers lazily when needed
        for (size_t i = 0; i < files.size(); i++) {
          files[i].parse_header();
          for (auto &kv : files[i].entries) {
            index.emplace(kv.first, std::make_pair((int)i, kv.second));
          }
        }
      }
    } else {
      // Load .safetensors files directly
      std::vector<std::filesystem::path> st_paths;
      for (auto &de : std::filesystem::directory_iterator(dir)) {
        if (!de.is_regular_file()) continue;
        auto p = de.path();
        if (p.extension() == ".safetensors") {
          st_paths.push_back(p);
        }
      }
      
      for (auto &p : st_paths) {
        LazySTFile f;
        f.path = p.string();
        f.mm = std::make_shared<LazyMMapFile>(f.path);
        f.parse_header();
        
        int fid = (int)files.size();
        for (auto &kv : f.entries) {
          index.emplace(kv.first, std::make_pair(fid, kv.second));
        }
        files.emplace_back(std::move(f));
      }
    }
    
    if (files.empty()) {
      throw std::runtime_error("no .safetensors in dir:" + dir);
    }
    
    std::cout << "[LazyLoader] Found " << files.size() << " safetensors files with " 
              << index.size() << " tensors (headers only)\n";
  }
  
  std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (const auto &kv : index) {
      out.push_back(kv.first);
    }
    return out;
  }
  
  TensorPtr make_tensor(const std::string &name, const std::shared_ptr<Context> &ctx, bool copy = false) {
    auto it = index.find(name);
    if (it == index.end()) return nullptr;
    
    int file_id = it->second.first;
    const auto &entry = it->second.second;
    auto &file = files[file_id];
    
    // Map file only when we need to read tensor data
    file.mm->map_file();
    auto* file_data = static_cast<const uint8_t*>(file.mm->get_data());
    const uint8_t* src = file_data + file.data_base + entry.file_offset;
    
    // Handle dtype cases
    const std::string &sd = entry.st_dtype;
    if (sd == "F32") {
      if (copy) {
        auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
        std::memcpy(T->data, src, entry.nbytes);
        file.mm->unmap();
        return T;
      }
      Tensor *raw = new Tensor(DType::FLOAT32, entry.shape);
      raw->data = const_cast<uint8_t *>(src);
      raw->ctx = ctx;
      std::shared_ptr<Tensor> T(raw, [file](Tensor *p) { delete p; });
      return T;
    } else if (sd == "F16") {
      if (copy) {
        auto T = Tensor::create(ctx, entry.shape, DType::FP16);
        std::memcpy(T->data, src, entry.nbytes);
        file.mm->unmap();
        return T;
      }
      Tensor *raw = new Tensor(DType::FP16, entry.shape);
      raw->data = const_cast<uint8_t *>(src);
      raw->ctx = ctx;
      std::shared_ptr<Tensor> T(raw, [file](Tensor *p) { delete p; });
      return T;
    } else if (sd == "I32") {
      if (copy) {
        auto T = Tensor::create(ctx, entry.shape, DType::INT32);
        std::memcpy(T->data, src, entry.nbytes);
        file.mm->unmap();
        return T;
      }
      Tensor *raw = new Tensor(DType::INT32, entry.shape);
      raw->data = const_cast<uint8_t *>(src);
      raw->ctx = ctx;
      std::shared_ptr<Tensor> T(raw, [file](Tensor *p) { delete p; });
      return T;
    } else if (sd == "I8") {
      if (copy) {
        auto T = Tensor::create(ctx, entry.shape, DType::INT8);
        std::memcpy(T->data, src, entry.nbytes);
        file.mm->unmap();
        return T;
      }
      Tensor *raw = new Tensor(DType::INT8, entry.shape);
      raw->data = const_cast<uint8_t *>(src);
      raw->ctx = ctx;
      std::shared_ptr<Tensor> T(raw, [file](Tensor *p) { delete p; });
      return T;
    } else if (sd == "BF16") {
      if (copy) {
        auto T = Tensor::create(ctx, entry.shape, DType::BF16);
        std::memcpy(T->data, src, entry.nbytes);
        file.mm->unmap();
        return T;
      }
      Tensor *raw = new Tensor(DType::BF16, entry.shape);
      raw->data = const_cast<uint8_t *>(src);
      raw->ctx = ctx;
      std::shared_ptr<Tensor> T(raw, [file](Tensor *p) { delete p; });
      return T;
    } else if (sd == "F8_E4M3") {
      auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
      size_t n = T->nelements();
      convert_fp8_e4m3_to_f32(src, reinterpret_cast<float *>(T->data), n);
      file.mm->unmap();
      return T;
    } else if (sd == "F8_E5M2") {
      auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
      size_t n = T->nelements();
      convert_fp8_e5m2_to_f32(src, reinterpret_cast<float *>(T->data), n);
      file.mm->unmap();
      return T;
    } else if (sd == "U8") {
      auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
      size_t n = T->nelements();
      convert_u8_to_f32(src, reinterpret_cast<float *>(T->data), n);
      file.mm->unmap();
      return T;
    } else if (sd == "BOOL") {
      auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
      size_t n = T->nelements();
      convert_bool_to_f32(src, reinterpret_cast<float *>(T->data), n);
      file.mm->unmap();
      return T;
    } else if (sd == "I16") {
      auto T = Tensor::create(ctx, entry.shape, DType::INT32);
      size_t n = T->nelements();
      convert_i16_to_i32(src, reinterpret_cast<int32_t *>(T->data), n);
      file.mm->unmap();
      return T;
    } else if (sd == "I64") {
      auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
      size_t n = T->nelements();
      convert_i64_to_f32(src, reinterpret_cast<float *>(T->data), n);
      file.mm->unmap();
      return T;
    } else if (sd == "F64") {
      auto T = Tensor::create(ctx, entry.shape, DType::FLOAT32);
      size_t n = T->nelements();
      convert_f64_to_f32(src, reinterpret_cast<float *>(T->data), n);
      file.mm->unmap();
      return T;
    } else {
      file.mm->unmap();
      throw std::runtime_error("unsupported dtype for ow::nn:" + sd);
    }
  }
  
  // FP8 E4M3: sign(1) exp(4, bias=7) mant(3). No Inf; exp=15 treated as NaN.
  static inline void convert_fp8_e4m3_to_f32(const uint8_t *src_bytes, float *dst,
                                             size_t n) {
    for (size_t i = 0; i < n; ++i) {
      uint8_t b = src_bytes[i];
      int sign = (b >> 7) & 0x1;
      int exp = (b >> 3) & 0xF;
      int mant = b & 0x7;
      float s = sign ? -1.0f : 1.0f;
      const int bias = 7;
      float v;
      if (exp == 0) {
        // subnormal or zero
        float m = mant / 8.0f;
        v = std::ldexp(m, 1 - bias);
      } else if (exp == 15) {
        // Treat as NaN (E4M3 typically has no Inf)
        v = std::numeric_limits<float>::quiet_NaN();
      } else {
        float m = 1.0f + mant / 8.0f;
        v = std::ldexp(m, exp - bias);
      }
      dst[i] = s * v;
    }
  }

  // FP8 E5M2: sign(1) exp(5, bias=15) mant(2). exp=31: Inf/NaN
  static inline void convert_fp8_e5m2_to_f32(const uint8_t *src_bytes, float *dst,
                                             size_t n) {
    for (size_t i = 0; i < n; ++i) {
      uint8_t b = src_bytes[i];
      int sign = (b >> 7) & 0x1;
      int exp = (b >> 2) & 0x1F;
      int mant = b & 0x3;
      float s = sign ? -1.0f : 1.0f;
      const int bias = 15;
      float v;
      if (exp == 0) {
        // subnormal or zero
        float m = mant / 4.0f;
        v = std::ldexp(m, 1 - bias);
      } else if (exp == 31) {
        if (mant == 0) {
          v = std::numeric_limits<float>::infinity();
        } else {
          v = std::numeric_limits<float>::quiet_NaN();
        }
      } else {
        float m = 1.0f + mant / 4.0f;
        v = std::ldexp(m, exp - bias);
      }
      dst[i] = s * v;
    }
  }

  static inline void convert_u8_to_f32(const uint8_t *src_bytes, float *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = float(src_bytes[i]);
    }
  }

  static inline void convert_bool_to_f32(const uint8_t *src_bytes, float *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = src_bytes[i] ? 1.0f : 0.0f;
    }
  }

  static inline void convert_i16_to_i32(const uint8_t *src_bytes, int32_t *dst, size_t n) {
    const int16_t *src = reinterpret_cast<const int16_t *>(src_bytes);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = int32_t(src[i]);
    }
  }

  static inline void convert_i64_to_f32(const uint8_t *src_bytes, float *dst, size_t n) {
    const int64_t *src = reinterpret_cast<const int64_t *>(src_bytes);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = float(src[i]);
    }
  }

  static inline void convert_f64_to_f32(const uint8_t *src_bytes, float *dst, size_t n) {
    const double *src = reinterpret_cast<const double *>(src_bytes);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = float(src[i]);
    }
  }
  

};

} // namespace ow::nn