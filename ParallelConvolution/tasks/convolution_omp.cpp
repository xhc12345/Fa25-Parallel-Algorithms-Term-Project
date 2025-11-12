#include "convolution_omp.h"
#include <omp.h>    // 1. Include OpenMP
#include <iomanip>  // For setting precision when printing time
#include <iostream>

void openmp_convolution(const std::vector<float>& input,
                        std::vector<float>& output,
                        const std::vector<float>& kernel,
                        int width,
                        int height,
                        int k_size) {
  int k_half = k_size / 2;

  // 2. Add the OpenMP pragma.
  // This tells OpenMP to split the 'y' loop iterations across all
  // available CPU threads. The 'x' loop and inner loops are
  // executed by that thread.
#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;

      for (int ky = -k_half; ky <= k_half; ++ky) {
        for (int kx = -k_half; kx <= k_half; ++kx) {
          int iy = y + ky;
          int ix = x + kx;

          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            int input_idx = iy * width + ix;
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
 * @brief Runs a single OpenMP benchmark.
 */
BenchmarkResult run_openmp_benchmark(
    const BenchmarkData& data,
    const std::vector<float>& expected_output) {
  std::cout << "  Running OpenMP CPU (" << omp_get_max_threads()
            << " threads)..." << std::endl;

  // Prepare output vector
  std::vector<float> output(data.width * data.height, 0.0f);

  // Start timer
  double start_time = omp_get_wtime();

  // Run convolution
  openmp_convolution(data.input, output, data.kernel, data.width, data.height,
                     data.k_size);

  // Stop timer
  double end_time = omp_get_wtime();
  double duration_ms = (end_time - start_time) * 1000.0;

  // Create result
  BenchmarkResult result;
  result.test_name = data.test_name;
  result.implementation_name = "OpenMP CPU";
  result.execution_time_ms = duration_ms;
  result.actual_output = output;

  // Verify correctness
  result.passed = verify_results(expected_output, result.actual_output);

  return result;
}