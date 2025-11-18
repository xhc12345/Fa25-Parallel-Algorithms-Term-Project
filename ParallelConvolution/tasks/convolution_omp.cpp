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

void openmp_convolution_avx(const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<float>& kernel,
    int width,
    int height,
    int k_size,
    int num_threads) {

    int k_half = k_size / 2;

    // Safe check: if image is too small, just run the safe path everywhere
    if (width <= k_size || height <= k_size) {
        // Fallback to single thread or simple parallel to avoid complexity
        // For benchmarking large images, this path is rarely taken.
        // (You can paste the original implementation here or just return)
        return;
    }

#pragma omp parallel for num_threads(num_threads)
    for (int y = 0; y < height; ++y) {

        // 1. Check if we are processing a Top or Bottom border row
        bool is_vertical_border = (y < k_half) || (y >= height - k_half);

        if (is_vertical_border) {
            // --- SLOW PATH (Safe with bounds checks) ---
            // Run for the entire width of the row
            for (int x = 0; x < width; ++x) {//!
                float sum = 0.0f;
                for (int ky = -k_half; ky <= k_half; ++ky) {//!
                    for (int kx = -k_half; kx <= k_half; ++kx) {//!
                        int iy = y + ky;
                        int ix = x + kx;
                        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                            sum += input[iy * width + ix] * kernel[(ky + k_half) * k_size + (kx + k_half)];
                        }
                    }
                }
                output[y * width + x] = sum;
            }
        }
        else {
            // --- FAST PATH (Center of the image) ---

            // Part A: Left Border (Safe Check)
            for (int x = 0; x < k_half; ++x) {//!
                float sum = 0.0f;
                for (int ky = -k_half; ky <= k_half; ++ky) {//!
                    for (int kx = -k_half; kx <= k_half; ++kx) {
                        int ix = x + kx; // We know y is safe, only check x
                        if (ix >= 0) {
                            sum += input[(y + ky) * width + ix] * kernel[(ky + k_half) * k_size + (kx + k_half)];//!
                        }
                    }
                }
                output[y * width + x] = sum;
            }

            // Part B: CENTER (The Optimized Loop)
            // We know y is safe. We know x is safe. NO IF CHECKS.
            // This is the hot loop where 90%+ of the time is spent.
            for (int x = k_half; x < width - k_half; ++x) {//!
                float sum = 0.0f;

                // 1. __restrict tells compiler: "These pointers do not overlap."
                //    This kills aliasing fears (Reason 1200/501).
                const float* __restrict input_ptr = input.data();
                const float* __restrict kernel_ptr = kernel.data();

                // 2. Pre-calculate constant base indices for the 'j' loop
                //    (Helps the compiler see linear memory access)
                for (int i = 0; i < k_size; ++i) {
                    int ky = i - k_half;
                    int input_row_start = (y + ky) * width;
                    int kernel_row_start = i * k_size;

                    // 3. The "Nuclear Option": Force vectorization with a reduction
                    //    This tells the compiler: "I promise it's safe to vector-sum this."
                    #pragma omp simd reduction(+:sum)
                    for (int j = 0; j < k_size; ++j) {
                        int kx = j - k_half;

                        // Access pattern is now: Base + j  (Perfect for AVX)
                        float val = input_ptr[input_row_start + (x + kx)];
                        float k_val = kernel_ptr[kernel_row_start + j];

                        sum += val * k_val;
                    }
                }
                output[y * width + x] = sum;
            }

            // Part C: Right Border (Safe Check)
            for (int x = width - k_half; x < width; ++x) {//!
                float sum = 0.0f;
                for (int ky = -k_half; ky <= k_half; ++ky) {//!
                    for (int kx = -k_half; kx <= k_half; ++kx) {
                        int ix = x + kx; // We know y is safe
                        if (ix < width) {
                            sum += input[(y + ky) * width + ix] * kernel[(ky + k_half) * k_size + (kx + k_half)];//!
                        }
                    }
                }
                output[y * width + x] = sum;
            }
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
    //openmp_convolution(data.input, output, data.kernel, data.width, data.height, data.k_size, num_threads_to_use);
    openmp_convolution_avx(data.input, output, data.kernel, data.width, data.height, data.k_size, num_threads_to_use);

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