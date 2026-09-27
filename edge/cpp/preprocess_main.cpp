// Write the model input for one image, for comparison with Python (edge/parity.sh).
//
// Usage: bloodcell_preprocess <image> <out.f32>
// out.f32 is the 1x3x224x224 float32 tensor bloodcell_infer feeds the model.

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#include "preprocess.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <image> <out.f32>\n";
    return 2;
  }
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "unreadable " << argv[1] << "\n";
    return 1;
  }
  std::vector<float> tensor(3 * kSize * kSize);
  preprocess(img, tensor.data());
  std::ofstream(argv[2], std::ios::binary)
      .write(reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(float));
  return 0;
}
