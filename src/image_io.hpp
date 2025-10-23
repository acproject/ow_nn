#pragma once

#include "tensor.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace ow::nn {

struct ImageLoadOptions {
  int target_h = 224;
  int target_w = 224;
  bool to_rgb = true;          // 输出 RGB
  bool keep_aspect = true;     // 短边对齐再中心裁剪
  bool center_crop = true;     // 居中裁剪到目标尺寸
  bool to_chw = true;          // 输出 CHW；否则 HWC
  bool normalize = true;       // /255 到 [0,1]
  bool add_batch_dim = false;  // 返回 [1,C,H,W]
};

// 读取图片并做基础前处理，输出 TensorPtr（默认 CHW, float32）
inline TensorPtr load_image_to_tensor(const std::shared_ptr<Context> &ctx,
                                      const std::string &path,
                                      const ImageLoadOptions &opt = {}) {
  if (!ctx) throw std::runtime_error("load_image_to_tensor: ctx is null");

  // 解码到 RGBA 8bit 缓冲
  int in_w = 0, in_h = 0, in_c = 4; // 固定 RGBA
  std::vector<unsigned char> rgba;

#if defined(__APPLE__) && defined(OW_WITH_APPLE_IMAGEIO)
  // Apple ImageIO 解码
  {
    // CoreFoundation / ImageIO
    #include <CoreFoundation/CoreFoundation.h>
    #include <ImageIO/ImageIO.h>
    #include <CoreGraphics/CoreGraphics.h>

    CFURLRef url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault,
                                                           reinterpret_cast<const UInt8 *>(path.c_str()),
                                                           path.size(), false);
    if (!url) throw std::runtime_error("ImageIO: invalid url");
    CGImageSourceRef src = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!src) throw std::runtime_error("ImageIO: cannot create source");
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!img) throw std::runtime_error("ImageIO: cannot decode image");
    in_w = (int)CGImageGetWidth(img);
    in_h = (int)CGImageGetHeight(img);
    if (in_w <= 0 || in_h <= 0) { CGImageRelease(img); throw std::runtime_error("ImageIO: invalid size"); }

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    size_t bytes_per_row = (size_t)in_w * 4;
    rgba.resize((size_t)in_w * (size_t)in_h * 4);
    CGContextRef ctxbmp = CGBitmapContextCreate(rgba.data(), in_w, in_h, 8, bytes_per_row, cs,
      kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!ctxbmp) { CGImageRelease(img); throw std::runtime_error("ImageIO: create bitmap ctx failed"); }
    CGRect rect = CGRectMake(0, 0, in_w, in_h);
    CGContextDrawImage(ctxbmp, rect, img);
    CGContextRelease(ctxbmp);
    CGImageRelease(img);
  }
#elif defined(_WIN32) && defined(OW_WITH_WIC)
  // Windows WIC 解码
  {
    #include <windows.h>
    #include <wincodec.h>

    auto utf8_to_wide = [](const std::string &s) -> std::wstring {
      if (s.empty()) return std::wstring();
      int sz = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
      if (sz <= 0) return std::wstring();
      std::wstring ws((size_t)sz - 1, L'\0');
      MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, ws.data(), sz);
      return ws;
    };

    bool need_uninit = false;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr)) need_uninit = true;
    else if (hr != RPC_E_CHANGED_MODE) throw std::runtime_error("WIC: CoInitializeEx failed");

    IWICImagingFactory *factory = nullptr;
    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                          IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) { if (need_uninit) CoUninitialize(); throw std::runtime_error("WIC: create factory failed"); }

    IWICBitmapDecoder *decoder = nullptr;
    std::wstring wpath = utf8_to_wide(path);
    hr = factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnDemand, &decoder);
    if (FAILED(hr) || !decoder) { factory->Release(); if (need_uninit) CoUninitialize(); throw std::runtime_error("WIC: open file failed"); }

    IWICBitmapFrameDecode *frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr) || !frame) { decoder->Release(); factory->Release(); if (need_uninit) CoUninitialize(); throw std::runtime_error("WIC: get frame failed"); }

    IWICFormatConverter *converter = nullptr;
    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr) || !converter) { frame->Release(); decoder->Release(); factory->Release(); if (need_uninit) CoUninitialize(); throw std::runtime_error("WIC: create converter failed"); }

    hr = converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
                               nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) { converter->Release(); frame->Release(); decoder->Release(); factory->Release(); if (need_uninit) CoUninitialize(); throw std::runtime_error("WIC: converter init failed"); }

    UINT w = 0, h = 0;
    hr = converter->GetSize(&w, &h);
    if (FAILED(hr) || w == 0 || h == 0) { converter->Release(); frame->Release(); decoder->Release(); factory->Release(); if (need_uninit) CoUninitialize(); throw std::runtime_error("WIC: invalid size"); }
    in_w = (int)w; in_h = (int)h;

    size_t stride = (size_t)in_w * 4;
    rgba.resize((size_t)in_h * stride);
    hr = converter->CopyPixels(nullptr, (UINT)stride, (UINT)rgba.size(), rgba.data());

    converter->Release(); frame->Release(); decoder->Release(); factory->Release();
    if (need_uninit) CoUninitialize();

    if (FAILED(hr)) throw std::runtime_error("WIC: copy pixels failed");
  }
#elif defined(OW_WITH_STB)
  {
    // 用户需在包含路径中提供 <stb_image.h>，实现由 stb_image_impl.cpp 提供
    #include <stb_image.h>
    int comp = 0;
    unsigned char *data = stbi_load(path.c_str(), &in_w, &in_h, &comp, 4);
    if (!data) throw std::runtime_error("stb_image: failed to load " + path);
    rgba.assign(data, data + ((size_t)in_w * (size_t)in_h * 4));
    stbi_image_free(data);
  }
#else
  throw std::runtime_error("No image loader enabled. Enable OW_WITH_APPLE_IMAGEIO on macOS, OW_WITH_WIC on Windows, or OW_WITH_STB with stb_image.h");
#endif

  // 从 RGBA 构造 RGB float 缓冲(HWC)
  auto to_float_rgb = [&](std::vector<float> &dst_rgb_hwc, bool normalize) {
    dst_rgb_hwc.resize((size_t)in_w * (size_t)in_h * 3);
    const float scale = normalize ? (1.0f / 255.0f) : 1.0f;
    for (int y = 0; y < in_h; ++y) {
      for (int x = 0; x < in_w; ++x) {
        size_t i_rgba = ((size_t)y * (size_t)in_w + (size_t)x) * 4;
        size_t i_rgb  = ((size_t)y * (size_t)in_w + (size_t)x) * 3;
        float r = (float)rgba[i_rgba + 0] * scale;
        float g = (float)rgba[i_rgba + 1] * scale;
        float b = (float)rgba[i_rgba + 2] * scale;
        if (!opt.to_rgb) {
          dst_rgb_hwc[i_rgb + 0] = b;
          dst_rgb_hwc[i_rgb + 1] = g;
          dst_rgb_hwc[i_rgb + 2] = r;
        } else {
          dst_rgb_hwc[i_rgb + 0] = r;
          dst_rgb_hwc[i_rgb + 1] = g;
          dst_rgb_hwc[i_rgb + 2] = b;
        }
      }
    }
  };

  std::vector<float> src_rgb_hwc;
  to_float_rgb(src_rgb_hwc, opt.normalize);

  // 计算缩放与裁剪
  int out_h = opt.target_h, out_w = opt.target_w;
  int resize_h = out_h, resize_w = out_w;
  if (opt.keep_aspect) {
    float scale = std::max((float)out_h / (float)in_h, (float)out_w / (float)in_w);
    resize_h = std::max(1, (int)std::round((float)in_h * scale));
    resize_w = std::max(1, (int)std::round((float)in_w * scale));
  }

  // 双线性缩放到 (resize_h, resize_w)
  auto bilinear_resize = [&](const std::vector<float> &src, int sh, int sw, std::vector<float> &dst, int dh, int dw) {
    dst.resize((size_t)dh * (size_t)dw * 3);
    const float rx = (float)sw / (float)dw;
    const float ry = (float)sh / (float)dh;
    for (int y = 0; y < dh; ++y) {
      float sy = (y + 0.5f) * ry - 0.5f;
      int y0 = (int)std::floor(sy);
      int y1 = std::min(y0 + 1, sh - 1);
      float wy = sy - (float)y0;
      y0 = std::max(0, std::min(y0, sh - 1));
      for (int x = 0; x < dw; ++x) {
        float sx = (x + 0.5f) * rx - 0.5f;
        int x0 = (int)std::floor(sx);
        int x1 = std::min(x0 + 1, sw - 1);
        float wx = sx - (float)x0;
        x0 = std::max(0, std::min(x0, sw - 1));
        size_t o = ((size_t)y * (size_t)dw + (size_t)x) * 3;
        for (int c = 0; c < 3; ++c) {
          auto S = [&](int yy, int xx) -> float {
            return src[((size_t)yy * (size_t)sw + (size_t)xx) * 3 + (size_t)c];
          };
          float v0 = (1 - wx) * S(y0, x0) + wx * S(y0, x1);
          float v1 = (1 - wx) * S(y1, x0) + wx * S(y1, x1);
          dst[o + (size_t)c] = (1 - wy) * v0 + wy * v1;
        }
      }
    }
  };

  std::vector<float> resized_rgb_hwc;
  if (resize_h != in_h || resize_w != in_w) bilinear_resize(src_rgb_hwc, in_h, in_w, resized_rgb_hwc, resize_h, resize_w);
  else resized_rgb_hwc = std::move(src_rgb_hwc);

  // 中心裁剪到 (out_h, out_w)
  std::vector<float> cropped_rgb_hwc;
  if (opt.keep_aspect && opt.center_crop && (resize_h > out_h || resize_w > out_w)) {
    int top = std::max(0, (resize_h - out_h) / 2);
    int left = std::max(0, (resize_w - out_w) / 2);
    cropped_rgb_hwc.resize((size_t)out_h * (size_t)out_w * 3);
    for (int y = 0; y < out_h; ++y) {
      for (int x = 0; x < out_w; ++x) {
        size_t di = ((size_t)y * (size_t)out_w + (size_t)x) * 3;
        size_t si = ((size_t)(y + top) * (size_t)resize_w + (size_t)(x + left)) * 3;
        cropped_rgb_hwc[di + 0] = resized_rgb_hwc[si + 0];
        cropped_rgb_hwc[di + 1] = resized_rgb_hwc[si + 1];
        cropped_rgb_hwc[di + 2] = resized_rgb_hwc[si + 2];
      }
    }
  } else {
    // 直接拉伸到目标（若尺寸不符）
    if (resize_h != out_h || resize_w != out_w) {
      bilinear_resize(resized_rgb_hwc, resize_h, resize_w, cropped_rgb_hwc, out_h, out_w);
    } else {
      cropped_rgb_hwc = std::move(resized_rgb_hwc);
    }
  }

  // 组装 Tensor（CHW 或 HWC；可选加 batch 维）
  if (opt.to_chw) {
    std::vector<int> shape = {3, out_h, out_w};
    if (opt.add_batch_dim) shape.insert(shape.begin(), 1); // [1,3,H,W]
    auto T = Tensor::create(ctx, shape, DType::FLOAT32);
    if (opt.add_batch_dim) {
      size_t hw = (size_t)out_h * (size_t)out_w;
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < out_h; ++y) {
          for (int x = 0; x < out_w; ++x) {
            float v = cropped_rgb_hwc[((size_t)y * (size_t)out_w + (size_t)x) * 3 + (size_t)c];
            size_t idx = 0 * (size_t)3 * hw + (size_t)c * hw + (size_t)y * (size_t)out_w + (size_t)x;
            T->set_from_float_flat(idx, v);
          }
        }
      }
    } else {
      size_t hw = (size_t)out_h * (size_t)out_w;
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < out_h; ++y) {
          for (int x = 0; x < out_w; ++x) {
            float v = cropped_rgb_hwc[((size_t)y * (size_t)out_w + (size_t)x) * 3 + (size_t)c];
            size_t idx = (size_t)c * hw + (size_t)y * (size_t)out_w + (size_t)x;
            T->set_from_float_flat(idx, v);
          }
        }
      }
    }
    return T;
  } else {
    std::vector<int> shape = {out_h, out_w, 3};
    if (opt.add_batch_dim) shape.insert(shape.begin(), 1); // [1,H,W,3]
    auto T = Tensor::create(ctx, shape, DType::FLOAT32);
    if (opt.add_batch_dim) {
      for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
          for (int c = 0; c < 3; ++c) {
            float v = cropped_rgb_hwc[((size_t)y * (size_t)out_w + (size_t)x) * 3 + (size_t)c];
            size_t idx = ((size_t)0 * (size_t)out_h + (size_t)y) * (size_t)out_w * 3 + (size_t)x * 3 + (size_t)c;
            T->set_from_float_flat(idx, v);
          }
        }
      }
    } else {
      for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
          for (int c = 0; c < 3; ++c) {
            float v = cropped_rgb_hwc[((size_t)y * (size_t)out_w + (size_t)x) * 3 + (size_t)c];
            size_t idx = ((size_t)y * (size_t)out_w + (size_t)x) * 3 + (size_t)c;
            T->set_from_float_flat(idx, v);
          }
        }
      }
    }
    return T;
  }
}

} // namespace ow::nn