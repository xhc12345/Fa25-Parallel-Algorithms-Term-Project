#include "convolution_ocl.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> // for std::setw/setfill

bool OpenCLContext::setup(const std::string& kernelPath, const std::string& kernelName) {
    cl_int err;
    cl_uint num_devices;

    // --- 1.1 Setup: Find platform, device, create context & queue ---
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to find any OpenCL platforms.\n";
        return false;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        std::cerr << "Error: Failed to find any GPU devices.\n";
        return false;
    }

    devices.resize(num_devices); // Resize the new vector
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL);
    // Now 'devices' vector contains all your GPUs

    // Create a context for ALL devices
    context = clCreateContext(NULL, num_devices, devices.data(), NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create OpenCL context.\n";
        return false;
    }

    // Create one queue for EACH device
    cl_command_queue_properties queueProps[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queues.resize(num_devices); // Resize the queues vector
    for (cl_uint i = 0; i < num_devices; ++i) {
        queues[i] = clCreateCommandQueueWithProperties(context, devices[i], queueProps, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to create command queue for device " << i << "\n";
            return false;
        }
    }

    // --- 1.2 Kernel & 1.3 Build: Create and build the program ---
    std::ifstream kernelFile(kernelPath);
    if (!kernelFile.is_open()) {
        std::cerr << "Error: Failed to open kernel file " << kernelPath << "\n";
        return false;
    }
    std::stringstream buffer;
    buffer << kernelFile.rdbuf();
    std::string kernelStr = buffer.str();
    if (kernelStr.empty()) {
        std::cerr << "Error: Kernel file " << kernelPath << " is empty!\n";
        return false;
    }

    const char* kernel_c_str = kernelStr.c_str();
    size_t kernel_len = kernelStr.length();
    program = clCreateProgramWithSource(context, 1, &kernel_c_str, &kernel_len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create program from source.\n";
        return false;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL build failed (error code " << err << ")\n";

        size_t log_size = 0;
        // Get the build log from the *first* device (logs are usually the same)
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        std::cerr << "Build log:\n" << log.data() << std::endl;
        return false;
    }

    kernel = clCreateKernel(program, kernelName.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to create kernel '" << kernelName << "'.\n";
        return false;
    }

    initialized = true;
    std::cout << "OpenCL context initialized successfully." << std::endl;
    return true;
}

void OpenCLContext::cleanup() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    for (cl_command_queue q : queues) { // Loop through all queues and release them
        if (q) clReleaseCommandQueue(q);
    }
    if (context) clReleaseContext(context);
    // Note: platform and device IDs don't need to be "released"
    initialized = false;
}

BenchmarkResult run_opencl_benchmark(
    OpenCLContext& ocl,
    cl_command_queue queue,
    const std::string& device_name,
    const BenchmarkData& data,
    const std::vector<float>& expected_output,
    int run_num, int total_runs) {

    std::string run_message = "Running OpenCL on " + device_name + "...";
    std::cout << "  [" << std::setw(2) << std::setfill('0') << std::right << run_num << "/"
        << std::setw(2) << total_runs << "]" << std::setfill(' ') // reset the fill character
        << " " << std::setw(30) << std::left << run_message
        << "\r" << std::flush;

    BenchmarkResult result;
    result.test_name = data.test_name;
    result.implementation_name = "OpenCL " + device_name;
    result.passed = false; // Default to fail

    if (!ocl.initialized) {
        std::cerr << "  Error: OpenCL context not initialized. Skipping test.\n";
        return result;
    }

    cl_int err;

    // --- 1.4 Memory (Host): Allocate and initialize host data ---
    size_t img_size_bytes = data.width * data.height * sizeof(float);
    size_t kernel_size_bytes = data.k_size * data.k_size * sizeof(float);
    std::vector<float> h_output(data.width * data.height, 0.0f);

    // --- 1.5 Memory (Device): Create buffers on the GPU ---
    cl_mem d_input = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, img_size_bytes, NULL, &err);
    cl_mem d_output = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, img_size_bytes, NULL, &err);
    cl_mem d_kernel = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, kernel_size_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "  Error: Failed to create OpenCL buffers.\n";
        return result;
    }

    // --- 1.6 Copy to Device: Write host data to device buffers ---
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, img_size_bytes, data.input.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_kernel, CL_TRUE, 0, kernel_size_bytes, data.kernel.data(), 0, NULL, NULL);

    // --- 1.7 Execute: Set kernel arguments and launch ---
    clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(ocl.kernel, 2, sizeof(cl_mem), &d_kernel);
    clSetKernelArg(ocl.kernel, 3, sizeof(int), &data.width);
    clSetKernelArg(ocl.kernel, 4, sizeof(int), &data.height);
    clSetKernelArg(ocl.kernel, 5, sizeof(int), &data.k_size);

    size_t global_work_size[2] = { (size_t)data.width, (size_t)data.height };
    cl_event kernelEvent;

    clEnqueueNDRangeKernel(queue, ocl.kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernelEvent);
    clWaitForEvents(1, &kernelEvent);

    // Measure time
    cl_ulong timeStart = 0, timeEnd = 0;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, nullptr);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, nullptr);
    double elapsed_ns = (timeEnd - timeStart);
    result.execution_time_ms = elapsed_ns / 1'000'000.0;

    // --- 1.8 Copy to Host: Read the result buffer back ---
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, img_size_bytes, h_output.data(), 0, NULL, NULL);

    // --- 1.9 Cleanup: Release buffers for this run ---
    clReleaseEvent(kernelEvent);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_kernel);

    // --- 1.10 Verify ---
    result.actual_output = h_output;
    result.passed = verify_results(expected_output, result.actual_output);

    return result;
}