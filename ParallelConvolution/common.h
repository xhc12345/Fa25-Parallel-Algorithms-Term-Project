#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>

/**
 * @struct BenchmarkData
 * @brief Holds all the data required to run a single convolution test case.
 */
struct BenchmarkData {
    std::string test_name;
    int width;
    int height;
    int k_size;
    std::vector<float> input;
    std::vector<float> kernel;

    // This will be populated by the sequential "golden" run
    std::vector<float> expected_output;
};

/**
 * @struct BenchmarkResult
 * @brief Holds the results from a single benchmark run.
 */
struct BenchmarkResult {
    std::string test_name;
    std::string implementation_name;
    double execution_time_ms;
    bool passed;
    std::vector<float> actual_output;
};

/**
 * @brief Verifies if two float vectors are element-wise equal within a tolerance.
 */
inline bool verify_results(const std::vector<float>& expected, const std::vector<float>& actual, float tolerance = 1e-5f) {
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i << ": expected "
                << expected[i] << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Prints a formatted summary table of all benchmark results.
 */
inline void print_summary(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(76, '=') << "\n";
    std::cout << "Benchmark Summary" << "\n";
    std::cout << std::string(76, '=') << "\n";

    // Headers
    std::cout << std::left
        << std::setw(25) << "Test Case"
        << std::setw(20) << "Implementation"
        << std::setw(15) << "Time (ms)"
        << std::setw(10) << "Passed"
        << "\n";
    std::cout << std::string(76, '-') << "\n";

    // Data
    for (const auto& res : results) {
        std::cout << std::left
            << std::setw(25) << res.test_name
            << std::setw(20) << res.implementation_name
            << std::fixed << std::setprecision(3) << std::setw(15) << res.execution_time_ms
            << std::setw(10) << (res.passed ? "Yes" : "NO")
            << "\n";
    }
    std::cout << std::string(76, '=') << "\n";
}