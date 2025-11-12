#include <iostream>
#include <string>
#include <vector>

// Include the implementations
#include "tasks/convolution_ocl.h"
#include "tasks/convolution_omp.h"
#include "tasks/convolution_seq.h"

// Include the helper for OpenCL platform checking
#include <CL/cl.h>
#include "tasks/SimpleTask.h"  // Assuming OpenCLTest is in here or similar

// Test if platform hardware and drivers satisfy OpenCL requirements
void OpenCLTest() {
  cl_uint num_platforms;
  cl_int err;

  // 1. Get the number of platforms
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    std::cerr << "Error: No OpenCL platforms found! (Error code: " << err << ")"
              << std::endl;
    std::cerr << "Please check your GPU drivers and OpenCL SDK installation."
              << std::endl;
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
    err =
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
      std::cout << "  No devices found on this platform." << std::endl;
      continue;
    }

    std::cout << "  Found " << num_devices << " device(s)." << std::endl;

    // 5. Get the device IDs
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
                   devices.data(), NULL);

    // 6. Iterate over each device and print its name
    for (cl_uint j = 0; j < num_devices; ++j) {
      char device_name[128];
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, device_name, NULL);
      std::cout << "    Device " << j << ": " << device_name << std::endl;
    }
  }

  std::cout << "\nOpenCL check complete." << std::endl;
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
  test3.test_name = "2048x2048 Box Blur";
  test3.width = 2048;
  test3.height = 2048;
  test3.k_size = 3;
  test3.input.assign(test3.width * test3.height, 10.0f);
  float v = 1.0f / 9.0f;
  test3.kernel = {v, v, v, v, v, v, v, v, v};
  tests.push_back(test3);
}

int main() {
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
  std::vector<BenchmarkResult> all_results;

  // 5. Run all benchmarks
  for (auto& test_data : test_cases) {
    std::cout << "\n--- Running Test: " << test_data.test_name << " ---"
              << std::endl;

    // 5a. Run Sequential (Golden Reference)
    BenchmarkResult seq_result = run_sequential_benchmark(test_data);
    test_data.expected_output =
        seq_result.actual_output;  // Save the golden result
    all_results.push_back(seq_result);

    // 5b. Run OpenMP
    BenchmarkResult omp_result =
        run_openmp_benchmark(test_data, test_data.expected_output);
    all_results.push_back(omp_result);

    // 5c. Run OpenCL
    if (ocl_ready) {
      BenchmarkResult ocl_result = run_opencl_benchmark(
          ocl_context, test_data, test_data.expected_output);
      all_results.push_back(ocl_result);
    } else {
      std::cout << "  Skipping OpenCL GPU (Context failed to initialize)."
                << std::endl;
    }
  }

  // 6. Print the final summary
  print_summary(all_results);

  // 7. Cleanup
  // ocl_context.cleanup() is called automatically by its destructor
  std::cout << "Benchmark suite finished." << std::endl;
  return 0;
}
