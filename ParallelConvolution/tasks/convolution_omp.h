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
 * @return A BenchmarkResult struct containing performance and correctness.
 */
BenchmarkResult run_openmp_benchmark(const BenchmarkData& data, const std::vector<float>& expected_output);