#pragma once
#include "../include/common.h"
#include "context.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <unordered_set>
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

// Dtype helpers
static inline size_t st_dtype_size(const std::string &s) {
  if (s=="F32") return 4; 
  if (s=="F16") return 2; 
  if (s=="BF16") return 2;
  if (s=="F64") return 8; 
  if (s=="I64") return 8; 
  if (s=="I32") return 4;
  if (s=="I16") return 2; 
  if (s=="I8") return 1; 
  if (s=="U8") return 1;
  if (s=="BOOL") return 1; 
  throw std::runtime_error("unsupported dtype:"+s);
}

static inline DType st_to_dtype(const std::string &s) {
  if (s=="F32") return DType::FLOAT32; 
  if (s=="F16") return DType::FP16;
  if (s=="I32") return DType::INT32; 
  if (s=="I8") return DType::INT8;
  // Extend as needed (BF16, etc.)
  throw std::runtime_error("unsupported dtype for ow::nn:"+s);
}

// Memory-mapped file abstraction
#ifdef _WIN32
struct MMapFile {
  std::wstring wpath; 
  HANDLE hFile=nullptr;
  HANDLE hMap=nullptr; 
  void* base=nullptr; 
  size_t size=0;
  static std::wstring to_w(const std::string &u8)
  {
    int n=MultiByteToWideChar(CP_UTF8,0,u8.c_str(),(int)u8.size(),NULL,0);
    std::wstring w;w.resize(n);
    MultiByteToWideChar(CP_UTF8,0,u8.c_str(),(int)u8.size(),w.data(),n);
    return w;
  }
  explicit MMapFile(const std::string &path){
    wpath=to_w(path); 
    hFile=CreateFileW(wpath.c_str(),GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL); 
    if(hFile==INVALID_HANDLE_VALUE) 
        throw std::runtime_error("open fail:"+path); 
    LARGE_INTEGER li; 
    if(!GetFileSizeEx(hFile,&li)) 
        throw std::runtime_error("size fail:"+path); 
    size=(size_t)li.QuadPart; 
    hMap=CreateFileMappingW(hFile,NULL,PAGE_READONLY,0,0,NULL); 
    if(!hMap) 
        throw std::runtime_error("map fail:"+path); 
    base=MapViewOfFile(hMap,FILE_MAP_READ,0,0,0); if(!base) throw std::runtime_error("map view fail:"+path); }
  ~MMapFile(){ if(base) UnmapViewOfFile(base); if(hMap) CloseHandle(hMap); if(hFile&&hFile!=INVALID_HANDLE_VALUE) CloseHandle(hFile);} 
};
#else
struct MMapFile { std::vector<uint8_t> buf; void* base=nullptr; size_t size=0; explicit MMapFile(const std::string &path){std::ifstream ifs(path, std::ios::binary); if(!ifs) throw std::runtime_error("open fail:"+path); ifs.seekg(0,std::ios::end); size_t s=(size_t)ifs.tellg(); ifs.seekg(0); buf.resize(s); ifs.read((char*)buf.data(), (std::streamsize)s); base=buf.data(); size=s; } };
#endif

struct STEntry { DType dtype; std::vector<int> shape; size_t nbytes; size_t file_offset; };
struct STFile { std::string path; std::shared_ptr<MMapFile> mm; uint64_t header_len=0; size_t data_base=0; std::unordered_map<std::string, STEntry> entries; };

// Parse header and build entries
static inline void parse_safetensors_header(STFile &f){
  if(f.mm->size<8) throw std::runtime_error("bad file:"+f.path);
  const uint8_t* p = reinterpret_cast<const uint8_t*>(f.mm->base);
  // little-endian 64-bit header length
  uint64_t N = *(reinterpret_cast<const uint64_t*>(p));
  f.header_len=N; if(8+N>f.mm->size) throw std::runtime_error("bad header len:"+f.path);
  std::string header((const char*)p+8, (size_t)N);
  json j = json::parse(header);
  f.data_base = 8 + (size_t)N;
  for(auto it=j.begin(); it!=j.end(); ++it){
    std::string name=it.key(); if(name=="__metadata__") continue;
    const auto &obj=it.value();
    std::string dtype = obj.at("dtype").get<std::string>();
    auto shape_j = obj.at("shape");
    auto offs = obj.at("data_offsets");
    size_t begin = offs.at(0).get<size_t>();
    size_t end = offs.at(1).get<size_t>();
    size_t nbytes = end-begin;
    std::vector<int> shape; shape.reserve(shape_j.size());
    for(auto &v: shape_j) shape.push_back(v.get<int>());
    size_t expect = st_dtype_size(dtype); for(int d:shape) expect*= (size_t)d;
    if(expect!=nbytes) throw std::runtime_error("size mismatch:"+name);
    STEntry e; e.dtype=st_to_dtype(dtype); e.shape=shape; e.nbytes=nbytes; e.file_offset=f.data_base+begin;
    f.entries.emplace(name,e);
  }
}

// Loader: load a directory with multiple .safetensors shards
struct SafetensorsLoader {
  std::vector<STFile> files;
  std::unordered_map<std::string, std::pair<int,STEntry>> index; // name -> (file_id, entry)

  void load_dir(const std::string &dir){
    files.clear(); index.clear();
    std::vector<std::filesystem::path> st_paths;
    for(auto &de: std::filesystem::directory_iterator(dir)){
      if(!de.is_regular_file()) continue;
      auto p=de.path();
      if(p.extension()==".safetensors"){
        st_paths.push_back(p);
      }
    }
    if(st_paths.empty()){
      // Try to parse HuggingFace index json: *.safetensors.index.json
      std::filesystem::path idx_path;
      for(auto &de: std::filesystem::directory_iterator(dir)){
        if(!de.is_regular_file()) continue;
        auto p = de.path();
        auto fname = p.filename().string();
        if(fname.size() >= 26 && fname.find(".safetensors.index.json") != std::string::npos){
          idx_path = p; break;
        }
      }
      if(!idx_path.empty()){
        std::ifstream ifs(idx_path.string());
        if(!ifs) throw std::runtime_error("open index fail:"+ idx_path.string());
        json j = json::parse(ifs);
        if(j.contains("weight_map")){
          std::unordered_set<std::string> shard_files;
          for(auto &kv: j["weight_map"].items()){
            std::string shard = kv.value().get<std::string>();
            shard_files.insert(shard);
          }
          std::vector<std::string> missing;
          for(const auto &s: shard_files){
            auto sp = (std::filesystem::path(dir) / s).string();
            if(!std::filesystem::exists(sp)) missing.push_back(sp);
          }
          if(!missing.empty()){
            std::string msg = "safetensors shards missing (" + std::to_string(missing.size()) + "): ";
            size_t lim = std::min<size_t>(missing.size(), 10);
            for(size_t i=0;i<lim;i++){ msg += missing[i]; msg += "; "; }
            throw std::runtime_error(msg + "Please download these shards into dir: " + dir);
          }
          // Load shards discovered from index
          for(const auto &s: shard_files){
            auto sp = (std::filesystem::path(dir) / s).string();
            STFile f; f.path=sp; f.mm=std::make_shared<MMapFile>(f.path);
            parse_safetensors_header(f);
            int fid=(int)files.size();
            for(auto &kv: f.entries) index.emplace(kv.first, std::make_pair(fid, kv.second));
            files.emplace_back(std::move(f));
          }
        }
      }
    } else {
      // Load directly from .safetensors files in directory
      for(auto &p: st_paths){
        STFile f; f.path=p.string(); f.mm=std::make_shared<MMapFile>(f.path);
        parse_safetensors_header(f);
        int fid=(int)files.size();
        for(auto &kv: f.entries) index.emplace(kv.first, std::make_pair(fid, kv.second));
        files.emplace_back(std::move(f));
      }
    }
    if(files.empty()) throw std::runtime_error("no .safetensors in dir:"+dir);
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out; out.reserve(index.size());
    for(auto &kv:index) out.push_back(kv.first);
    return out;
  }

  TensorPtr make_tensor(const std::string &name, const std::shared_ptr<Context> &ctx, bool copy=false){
    auto it=index.find(name); if(it==index.end()) throw std::runtime_error("tensor not found:"+name);
    int fid=it->second.first; const STEntry &e=it->second.second; const STFile &f=files[fid];
    const uint8_t* src = reinterpret_cast<const uint8_t*>(f.mm->base)+ e.file_offset;
    if(copy){ auto T = Tensor::create(ctx, e.shape, e.dtype); std::memcpy(T->data, src, e.nbytes); return T; }
    // 零拷贝视图：保持 mmap 文件生命周期 + 正确删除临时 Tensor
    Tensor *raw = new Tensor(e.dtype, e.shape);
    raw->data = const_cast<uint8_t*>(src);
    raw->ctx = ctx;
    auto owner = f.mm; // 捕获以保持映射有效
    std::shared_ptr<Tensor> T(raw, [owner](Tensor *p){ delete p; });
    return T;
  }

  std::unordered_map<std::string, TensorPtr> load_all(const std::shared_ptr<Context> &ctx, bool copy=false){
    std::unordered_map<std::string, TensorPtr> out; out.reserve(index.size());
    for(auto &kv:index) out.emplace(kv.first, make_tensor(kv.first, ctx, copy));
    return out;
  }
};

} // namespace ow::nn