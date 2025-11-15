#pragma once

#include <vector>
#include <string>
#include "../common.h"

// Include the correct OpenCL header for your system
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

/**
 * @struct OpenCLContext
 * @brief Manages all persistent OpenCL objects (platform, device, context, etc.)
 */
struct OpenCLContext {
    cl_platform_id platform = NULL;
    std::vector<cl_device_id> devices;
    cl_context context = NULL;
    std::vector<cl_command_queue> queues;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    bool initialized = false;

    /**
     * @brief Initializes OpenCL platform, device, context, queue,
     * and builds the kernel program.
     * @param kernelPath Path to the .cl kernel file.
     * @param kernelName The name of the kernel function (e.g., "convolve_fp32").
     * @return true on success, false on failure.
     */
    bool setup(const std::string& kernelPath, const std::string& kernelName);

    /**
     * @brief Releases all allocated OpenCL resources.
     */
    void cleanup();

    // Destructor ensures cleanup is called
    ~OpenCLContext() {
        cleanup();
    }
};

/**
 * @brief Runs a single OpenCL benchmark.
 * @param ocl The initialized OpenCLContext.
 * @param data The benchmark data to run.
 * @param expected_output The "golden" output to verify against.
 * @param run_num The current iteration number (e.g., 1).
 * @param total_runs The total number of iterations (e.g., 10).
 * @return A BenchmarkResult struct containing performance and correctness.
 */
BenchmarkResult run_opencl_benchmark(
    OpenCLContext& ocl,
    cl_command_queue queue,
    const std::string& device_name,
    const BenchmarkData& data,
    const std::vector<float>& expected_output,
    int run_num, int total_runs);