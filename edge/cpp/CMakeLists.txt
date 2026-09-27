cmake_minimum_required(VERSION 3.18)
project(bloodcell_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Point ONNXRUNTIME_ROOT at an extracted onnxruntime-linux-aarch64-<ver> release
# (64-bit Raspberry Pi OS) or onnxruntime-linux-x64-<ver> (desktop), e.g.
#   cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-aarch64-1.30.0
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime release")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=<path to onnxruntime release>")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib REQUIRED NO_DEFAULT_PATH)

add_executable(bloodcell_infer main.cpp)
target_include_directories(bloodcell_infer PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(bloodcell_infer PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})
set_target_properties(bloodcell_infer PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)

# Writes the model input for one image, so edge/parity.sh can compare it with Python's
add_executable(bloodcell_preprocess preprocess_main.cpp)
target_link_libraries(bloodcell_preprocess PRIVATE ${OpenCV_LIBS})
