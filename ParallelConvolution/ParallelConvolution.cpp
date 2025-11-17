#include <iostream>
#include <string>
#include <vector>
#include <functional> // For std::function
#include <numeric>    // For std::accumulate
#include <iomanip>    // For std::setw/setfill

// Include the implementations
#include "tasks/convolution_ocl.h"
#include "tasks/convolution_omp.h"
#include "tasks/convolution_seq.h"

#include <omp.h>    // Include OpenMP

// Include the helper for OpenCL platform checking
#include <CL/cl.h>

std::string get_opencl_device_name(cl_device_id device_id) {
    if (device_id == NULL) {
        return "Unknown Device";
    }

    // First, get the size of the name string
    size_t name_size = 0;
    cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &name_size);
    if (err != CL_SUCCESS || name_size == 0) {
        return "Unknown Device";
    }

    // Now, allocate space and get the name
    std::vector<char> name_buffer(name_size);
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, name_size, name_buffer.data(), NULL);
    if (err != CL_SUCCESS) {
        return "Unknown Device";
    }

    // Convert to a C++ string; drop null terminator at the end
    std::string device_name = std::string(name_buffer.begin(), name_buffer.end() - 1);
    if      (device_name == "gfx1032")  device_name = "NAVI 23";
    else if (device_name == "gfx1035")  device_name = "Radeon 680M";
    return  device_name;
}

// Test if platform hardware and drivers satisfy OpenCL requirements
void OpenCLTest() {
  cl_uint num_platforms;
  cl_int err;

  // 1. Get the number of platforms
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    std::cerr << "Error: No OpenCL platforms found! (Error code: " << err << ")" << std::endl;
    std::cerr << "Please check your GPU drivers and OpenCL SDK installation." << std::endl;
    exit(-1);
  }

  std::cout << "Found " << num_platforms << " OpenCL platform(s)." << std::endl;

  // 2. Get the platform IDs
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), NULL);

  // 3. Iterate over each platform
  for (cl_uint i = 0; i < num_platforms; ++i) {
    char platform_name[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platform_name, NULL);
    std::cout << "\n--- Platform " << i << ": " << platform_name << " ---"
              << std::endl;

    // 4. Get the number of devices on this platform
    cl_uint num_devices;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
      std::cout << "\tNo devices found on this platform." << std::endl;
      continue;
    }

    std::cout << "\tFound " << num_devices << " device(s)." << std::endl;

    // 5. Get the device IDs
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
                   devices.data(), NULL);

    // 6. Iterate over each device and print its name
    for (cl_uint j = 0; j < num_devices; ++j) {
        std::string device_name = get_opencl_device_name(devices[j]);
        std::cout << "\t\tDevice " << j << ": " << device_name << std::endl;
    }
  }

  std::cout << "\nOpenCL check complete." << std::endl;
}

/**
 * @brief Runs a given benchmark function N times, averages the results,
 * and returns the result from the *last* run (with averaged time).
 *
 * @param num_runs The number of times to execute the benchmark.
 * @param benchmark_func A C++ lambda that now takes (run_num, total_runs)
 * and returns a BenchmarkResult.
 * @return A BenchmarkResult. 'execution_time_ms' is the average time.
 * 'actual_output' and 'passed' are from the final run.
 */
BenchmarkResult run_and_average_benchmark(
    int num_runs,
    std::function<BenchmarkResult(int run_num, int total_runs)> benchmark_func)
{
    if (num_runs <= 0) num_runs = 1; // Safety check

    double total_time_ms = 0.0;
    BenchmarkResult last_result;

    for (int curr_run = 1; curr_run <= num_runs; ++curr_run) {
        last_result = benchmark_func(curr_run, num_runs);
        total_time_ms += last_result.execution_time_ms;

        // If any run fails, stop immediately and return the failed result
        if (!last_result.passed) {
            std::cout << std::endl; // Move to a new line after the \r
            std::cerr << "  ! Benchmark failed on run " << curr_run << ". Stopping averaging." << std::endl;
            return last_result;
        }
    }

    // All runs passed, return the last result, but overwrite the time with the average.
    double average_runtime = total_time_ms / (double)num_runs;
    std::cout << std::endl << "\t> Average runtime:\t" << average_runtime << " ms" << std::endl;
    last_result.execution_time_ms = average_runtime;
    return last_result;
}

/**
 * @brief Creates a list of standard benchmark test cases.
 */
void create_test_cases(std::vector<BenchmarkData>& tests) {
  // --- Test Case 1: 5x5 Edge Detect ---
  BenchmarkData test1;
  test1.test_name = "5x5 Edge Detect";
  test1.width = 5;
  test1.height = 5;
  test1.k_size = 3;
  test1.input = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  test1.kernel = {0, -1, 0, -1, 5, -1, 0, -1, 0};
  tests.push_back(test1);

  // --- Test Case 2: 1024x1024 Identity ---
  BenchmarkData test2;
  test2.test_name = "1024x1024 Identity";
  test2.width = 1024;
  test2.height = 1024;
  test2.k_size = 3;
  test2.input.assign(test2.width * test2.height, 1.0f);
  test2.kernel = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  tests.push_back(test2);

  // --- Test Case 3: 2048x2048 Box Blur ---
  BenchmarkData test3;
  test3.test_name = "2048x2048 Box Blur (3x3)";
  test3.width = 2048;
  test3.height = 2048;
  test3.k_size = 3;
  test3.input.assign(test3.width * test3.height, 10.0f);
  float v = 1.0f / 9.0f;
  test3.kernel = {v, v, v, v, v, v, v, v, v};
  tests.push_back(test3);

  // --- Test Case 4: 1024x1024 Compute-Heavy Blur ---
  BenchmarkData test4;
  test4.test_name = "1024x1024 Big Kernel";
  test4.width = 1024;
  test4.height = 1024;
  test4.k_size = 9; // <-- This is the huge kernel
  test4.input.assign(test4.width * test4.height, 5.0f);
  // Create a normalized box blur kernel
  int k_total = test4.k_size * test4.k_size;
  float v_large = 1.0f / (float)k_total;
  test4.kernel.assign(k_total, v_large);
  tests.push_back(test4);
}

int main() {
    #if defined(_OPENMP)
        std::cout << "OpenMP is enabled." << std::endl;
    #else
        std::cout << "OpenMP is NOT enabled." << std::endl;
    #endif
    std::cout << "Running OpenMP with " << omp_get_max_threads() << " CPU threads." << std::endl;

    // 1. Check OpenCL setup
    OpenCLTest();

    // 2. Setup OpenCL context once
    OpenCLContext ocl_context;
    // Note: Adjust path/kernel name if different
    bool ocl_ready = ocl_context.setup("kernels/convolution.cl", "convolve_fp32");

    // 3. Define all our test cases
    std::vector<BenchmarkData> test_cases;
    create_test_cases(test_cases);

    // 4. Store all results
    int max_threads = omp_get_max_threads();
    std::vector<int> thread_configs = { 1, 2, 4, 8, 12, 16 };
    std::vector<BenchmarkResult> all_results;
    const int NUM_RUNS = 10;
    std::cout << "\nEach benchmark will be run " << NUM_RUNS
        << " times and averaged." << std::endl;

    // 5. Run all benchmarks
    for (auto& test_data : test_cases) {
        std::cout << "\n--- Running Test: " << test_data.test_name << " ---" << std::endl;

        // 5a. Run Sequential (Golden Reference)
        // We use a lambda [&] to "capture" the variables needed by the function
        BenchmarkResult seq_result = run_and_average_benchmark(NUM_RUNS,
            [&](int r, int t) {
                return run_sequential_benchmark(test_data, r, t);
            }
        );
        test_data.expected_output = seq_result.actual_output;
        all_results.push_back(seq_result);

        // 5b. Run OpenMP through all thread configs
        for (int thread_count : thread_configs) {

            // Skip this test if it requests more threads than available
            if (thread_count > max_threads) {
                std::cout << "  Skipping " << thread_count
                    << "T (max is " << max_threads << ")" << std::endl;
                continue;
            }

            BenchmarkResult result = run_and_average_benchmark(NUM_RUNS,
                [&](int r, int t) {
                    return run_openmp_benchmark(test_data, test_data.expected_output, r, t, thread_count);
                }
            );
            all_results.push_back(result);
        }

        // 5c. Run OpenCL
        if (ocl_ready) {
            // Go through each device
            for (int i = 0; i < ocl_context.queues.size(); ++i) {
                std::string ocl_device_name = get_opencl_device_name(ocl_context.devices[i]);
                BenchmarkResult ocl_result = run_and_average_benchmark(NUM_RUNS,
                    [&](int r, int t) {
                        return run_opencl_benchmark(
                            ocl_context,
                            ocl_context.queues[i],
                            ocl_device_name,
                            test_data,
                            test_data.expected_output,
                            r, t);
                    }
                );
                all_results.push_back(ocl_result);
            }
        }
        else {
            std::cout << "  Skipping OpenCL GPU (Context failed to initialize)." << std::endl;
        }
    }

    // 6. Print the final summary
    print_summary(all_results);

    // 7. Cleanup
    // ocl_context.cleanup() is called automatically by its destructor
    std::cout << "Benchmark suite finished." << std::endl;
    return 0;
}
