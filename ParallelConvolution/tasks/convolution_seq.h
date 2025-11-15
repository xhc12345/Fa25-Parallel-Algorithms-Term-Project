#pragma once

#include <vector>
#include "../common.h" // Include the common definitions

/**
 * @brief Performs a 2D convolution sequentially on the CPU.
 */
void sequential_convolution(const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<float>& kernel,
    int width, int height, int k_size);

/**
 * @brief Runs a single sequential benchmark.
 * @param data The benchmark data to run.
 * @param run_num The current iteration number (e.g., 1).
 * @param total_runs The total number of iterations (e.g., 10).
 * @return A BenchmarkResult struct containing performance and output.
 */
BenchmarkResult run_sequential_benchmark(const BenchmarkData& data,
    int run_num, int total_runs); // <-- MODIFIED