// Preprocessing shared by bloodcell_infer and bloodcell_preprocess. It must match
// src/bloodcell/model.py and the <model>.json sidecar (ADR 0002): BGR->RGB,
// resize to 224x224 exactly as Pillow does, /255, normalize with ImageNet
// mean/std, NCHW float32.
#pragma once

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int kSize = 224;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

// One axis of Pillow's bilinear resample (Resample.c), in its 8-bit fixed point.
// When shrinking, the triangle filter widens with the scale, so every input
// pixel counts (antialiasing). OpenCV's INTER_LINEAR samples only the nearest
// two and gives up to 37/255 different values on PBC images (#35).
struct Taps {
  static constexpr int kBits = 22;
  int ksize = 0;
  std::vector<int> first, count;
  std::vector<int32_t> weight;  // ksize per output pixel

  Taps(int in, int out) {
    const double scale = double(in) / out, support = std::max(scale, 1.0), inv = 1.0 / support;
    ksize = int(std::ceil(support)) * 2 + 1;
    first.resize(out);
    count.resize(out);
    weight.assign(size_t(out) * ksize, 0);
    std::vector<double> w(ksize);
    for (int i = 0; i < out; ++i) {
      const double center = (i + 0.5) * scale;
      const int a = std::max(int(center - support + 0.5), 0);
      const int n = std::min(int(center + support + 0.5), in) - a;
      double sum = 0;
      for (int j = 0; j < n; ++j) {
        const double t = std::abs((a + j - center + 0.5) * inv);
        sum += w[j] = t < 1 ? 1 - t : 0;
      }
      for (int j = 0; j < n; ++j)
        weight[size_t(i) * ksize + j] = int32_t(std::lround(w[j] / sum * (1 << kBits)));
      first[i] = a;
      count[i] = n;
    }
  }

  // Weighted sum of n inputs spaced `stride` bytes apart, rounded and clipped.
  uint8_t apply(int i, const uint8_t* src, size_t stride) const {
    int64_t s = int64_t(1) << (kBits - 1);
    const int32_t* k = &weight[size_t(i) * ksize];
    for (int j = 0; j < count[i]; ++j) s += int64_t(src[(first[i] + j) * stride]) * k[j];
    return uint8_t(std::clamp<int64_t>(s >> kBits, 0, 255));
  }
};

// Resize like torchvision's Resize on a PIL image: horizontal pass, then vertical.
inline cv::Mat pil_resize(const cv::Mat& src, int width, int height) {
  const Taps h(src.cols, width), v(src.rows, height);
  cv::Mat tmp(src.rows, width, CV_8UC3), dst(height, width, CV_8UC3);
  for (int y = 0; y < src.rows; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) tmp.ptr<uint8_t>(y)[3 * x + c] = h.apply(x, src.ptr<uint8_t>(y) + c, 3);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) dst.ptr<uint8_t>(y)[3 * x + c] = v.apply(y, tmp.ptr<uint8_t>(0) + 3 * x + c, tmp.step);
  return dst;
}

inline void preprocess(const cv::Mat& bgr, float* dst) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  const cv::Mat resized = pil_resize(rgb, kSize, kSize);
  const int plane = kSize * kSize;
  for (int y = 0; y < kSize; ++y) {
    const auto* row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < kSize; ++x)
      for (int c = 0; c < 3; ++c)
        dst[c * plane + y * kSize + x] = (row[x][c] / 255.0f - kMean[c]) / kStd[c];
  }
}
