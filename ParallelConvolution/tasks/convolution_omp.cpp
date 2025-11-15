#include "convolution_omp.h"
#include <omp.h>    // 1. Include OpenMP
#include <iomanip>  // For setting precision when printing time
#include <iostream>

#define TILE_SIZE 32 // This is the "block size". You must tune this!

#if defined(_OPENMP)
    #pragma message("OpenMP is enabled.")
#else
    #pragma message("OpenMP is NOT enabled.")
#endif

/*
void openmp_convolution(const std::vector<float>& input,
                        std::vector<float>& output,
                        const std::vector<float>& kernel,
                        int width,
                        int height,
                        int k_size) {
  int k_half = k_size / 2;

  // 1. Add OpenMP pragma to the *outer* tile loop
  // We parallelize the tiles, not the inner rows
#pragma omp parallel for
  for (int tile_y = 0; tile_y < height; tile_y += TILE_SIZE) {
      for (int tile_x = 0; tile_x < width; tile_x += TILE_SIZE) {

          // --- These are your ORIGINAL loops ---
          // But now they are "clamped" to only run inside the tile
          for (int y = tile_y; y < std::min(tile_y + TILE_SIZE, height); ++y) {
              for (int x = tile_x; x < std::min(tile_x + TILE_SIZE, width); ++x) {

                  float sum = 0.0f;
                  for (int ky = -k_half; ky <= k_half; ++ky) {
                      for (int kx = -k_half; kx <= k_half; ++kx) {

                          int iy = y + ky;
                          int ix = x + kx;

                          // (Boundary checks remain the same)
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
  }
}
*/

void openmp_convolution(const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<float>& kernel,
    int width,
    int height,
    int k_size,
    int num_threads) {
    int k_half = k_size / 2;

    // 2. Add the OpenMP pragma.
    // This tells OpenMP to split the 'y' loop iterations across all
    // available CPU threads. The 'x' loop and inner loops are
    // executed by that thread.
#pragma omp parallel for num_threads(num_threads)
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
    const std::vector<float>& expected_output,
    int run_num, int total_runs,
    int num_threads_to_use) {
    if (num_threads_to_use < 1) num_threads_to_use = 1;
    else if (num_threads_to_use > omp_get_max_threads()) num_threads_to_use = omp_get_max_threads();

    std::string label = "Running OpenMP CPU (" +
        std::to_string(num_threads_to_use) + " threads)...";
    std::cout << "  [" << std::setw(2) << std::setfill('0') << std::right << run_num << "/"
        << std::setw(2) << total_runs << "] " << std::setfill(' ') // reset the fill character
        << std::setw(30) << std::left << label
        << "\r" << std::flush;

    // Prepare output vector
    std::vector<float> output(data.width * data.height, 0.0f);

    // Start timer
    double start_time = omp_get_wtime();

    // Run convolution
    openmp_convolution(data.input, output, data.kernel, data.width, data.height,
                        data.k_size, num_threads_to_use);

    // Stop timer
    double end_time = omp_get_wtime();
    double duration_ms = (end_time - start_time) * 1000.0;

    // Create result
    BenchmarkResult result;
    result.test_name = data.test_name;
    result.implementation_name = "OpenMP CPU (" + std::to_string(num_threads_to_use)+"T)";
    result.execution_time_ms = duration_ms;
    result.actual_output = output;

    // Verify correctness
    result.passed = verify_results(expected_output, result.actual_output);

    return result;
}