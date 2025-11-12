#include "convolution_seq.h"
#include <chrono>   // 1. Include for C++ timing
#include <iomanip>  // 2. Include for output formatting
#include <iostream>

void sequential_convolution(const std::vector<float>& input,
                            std::vector<float>& output,
                            const std::vector<float>& kernel,
                            int width,
                            int height,
                            int k_size) {
  int k_half = k_size / 2;

  // Iterate over each output pixel (y, x)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;

      // Iterate over the kernel (ky, kx)
      for (int ky = -k_half; ky <= k_half; ++ky) {
        for (int kx = -k_half; kx <= k_half; ++kx) {
          int iy = y + ky;
          int ix = x + kx;

          // Simple "clamp-to-edge" padding
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            // (y, x) in 2D -> (y * width + x) in 1D
            int input_idx = iy * width + ix;
            // (ky, kx) in 2D -> ((ky + k_half) * k_size + (kx + k_half)) in 1D
            int kernel_idx = (ky + k_half) * k_size + (kx + k_half);
            sum += input[input_idx] * kernel[kernel_idx];
          }
        }
      }
      output[y * width + x] = sum;
    }
  }
}

/**
 * @brief Runs a single sequential benchmark.
 */
BenchmarkResult run_sequential_benchmark(const BenchmarkData& data) {
  std::cout << "  Running Sequential CPU..." << std::endl;

  // Prepare output vector
  std::vector<float> output(data.width * data.height, 0.0f);

  // Start timer
  auto start = std::chrono::high_resolution_clock::now();

  // Run convolution
  sequential_convolution(data.input, output, data.kernel, data.width,
                         data.height, data.k_size);

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Create result
  BenchmarkResult result;
  result.test_name = data.test_name;
  result.implementation_name = "Sequential CPU";
  result.execution_time_ms = duration.count() / 1000.0;
  result.actual_output = output;

  // This is the "golden" reference, so it "passes" by definition.
  // The main runner will set this to true.
  result.passed = true;

  return result;
}