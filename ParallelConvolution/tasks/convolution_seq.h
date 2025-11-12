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
 * @return A BenchmarkResult struct containing performance and output.
 * Note: 'passed' field is not set, as this is the reference.
 */
BenchmarkResult run_sequential_benchmark(const BenchmarkData& data);