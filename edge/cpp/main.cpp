// Classify blood cell images with an exported ONNX model (ONNX Runtime C++ + OpenCV).
//
// Usage: bloodcell_infer <model.onnx> <image_or_dir> [--threads N] [--warmup N]
//
// Preprocessing (preprocess.hpp) must match src/bloodcell/model.py.
// Class names and the temperature come from the sidecar JSON: confidence is
// softmax(logits / temperature), calibrated as in Python (ADR 0005).

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "preprocess.hpp"

namespace fs = std::filesystem;

// Minimal extraction of "classes": [...] from the sidecar; avoids a JSON dependency.
static std::vector<std::string> load_classes(const fs::path& json_path) {
  std::ifstream f(json_path);
  if (!f) return {};
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  std::smatch block;
  if (!std::regex_search(s, block, std::regex(R"("classes"\s*:\s*\[([^\]]*)\])"))) return {};
  std::vector<std::string> out;
  std::string body = block[1];
  std::regex item(R"re("([^"]*)")re");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), item); it != std::sregex_iterator(); ++it)
    out.push_back((*it)[1]);
  return out;
}

// "temperature": <number> from the sidecar; 1 (uncalibrated) when it's missing.
static float load_temperature(const fs::path& json_path) {
  std::ifstream f(json_path);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  std::smatch m;
  if (!std::regex_search(s, m, std::regex(R"("temperature"\s*:\s*([0-9.eE+-]+))"))) return 1.0f;
  return std::stof(m[1].str());
}


static std::vector<fs::path> list_images(const fs::path& p) {
  if (fs::is_regular_file(p)) return {p};
  std::vector<fs::path> out;
  for (const auto& e : fs::recursive_directory_iterator(p)) {
    auto ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <model.onnx> <image_or_dir> [--threads N] [--warmup N]\n";
    return 2;
  }
  fs::path model_path = argv[1], input = argv[2];
  int threads = 4, warmup = 5;
  for (int i = 3; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--threads") threads = std::stoi(argv[i + 1]);
    else if (flag == "--warmup") warmup = std::stoi(argv[i + 1]);
  }

  auto sidecar = fs::path(model_path).replace_extension(".json");
  auto classes = load_classes(sidecar);
  const float temperature = load_temperature(sidecar);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bloodcell");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(env, model_path.c_str(), opts);

  Ort::AllocatorWithDefaultOptions alloc;
  auto in_name = session.GetInputNameAllocated(0, alloc);
  auto out_name = session.GetOutputNameAllocated(0, alloc);
  const char* in_names[] = {in_name.get()};
  const char* out_names[] = {out_name.get()};

  std::vector<float> tensor(3 * kSize * kSize);
  std::array<int64_t, 4> shape{1, 3, kSize, kSize};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  auto run = [&]() {
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(), shape.data(), shape.size());
    return session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  };

  auto images = list_images(input);
  if (images.empty()) {
    std::cerr << "no images found at " << input << "\n";
    return 1;
  }

  for (int i = 0; i < warmup; ++i) run();

  std::vector<double> ms;
  ms.reserve(images.size());
  for (const auto& path : images) {
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "skip unreadable " << path << "\n";
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    preprocess(img, tensor.data());
    auto out = run();
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    const float* logits = out[0].GetTensorData<float>();
    size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float mx = *std::max_element(logits, logits + n), sum = 0.f;
    for (size_t k = 0; k < n; ++k) sum += std::exp((logits[k] - mx) / temperature);
    size_t best = std::max_element(logits, logits + n) - logits;
    float conf = 1.0f / sum;  // exp((max - max) / T) / sum
    std::string label = best < classes.size() ? classes[best] : std::to_string(best);
    std::cout << path.string() << "\t" << label << "\t" << conf << "\n";
  }

  if (!ms.empty()) {
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double q) { return ms[std::min(ms.size() - 1, size_t(q * ms.size()))]; };
    double total = 0;
    for (double v : ms) total += v;
    std::cerr << "images " << ms.size() << "  p50 " << pct(0.50) << " ms  p95 " << pct(0.95)
              << " ms  throughput " << 1000.0 * ms.size() / total << " img/s (threads " << threads << ")\n";
  }
  return 0;
}
