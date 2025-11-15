#pragma once

#include <vector>
#include "../common.h" // Include the common definitions

/**
 * @brief Performs a 2D convolution in parallel on the CPU using OpenMP.
 */
void openmp_convolution(const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<float>& kernel,
    int width, int height, int k_size);

/**
 * @brief Runs a single OpenMP benchmark.
 * @param data The benchmark data to run.
 * @param expected_output The "golden" output to verify against.
 * @param run_num The current iteration number (e.g., 1).
 * @param total_runs The total number of iterations (e.g., 10).
 * @return A BenchmarkResult struct containing performance and correctness.
 */
BenchmarkResult run_openmp_benchmark(const BenchmarkData& data,
    const std::vector<float>& expected_output,
    int run_num, int total_runs); // <-- MODIFIED