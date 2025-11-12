#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// On Windows/Linux: #include <CL/cl.h>
// On macOS: #include <OpenCL/opencl.h>
#include <CL/cl.h>


int ConvolutionParallelGPUTest1() {
    // 1. Host (C++) Code
    cl_int err;

    // --- 1.1 Setup: Find platform, device, create context & queue ---
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL); // (Using first platform found)

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // (Using first GPU found)

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue_properties queueProps[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queueProps, &err);

    // --- 1.2 Kernel & 1.3 Build: Create and build the program ---
    std::string kernelPath = "kernels/convolution.cl";
    std::ifstream kernelFile(kernelPath);
    if (!kernelFile.is_open()) {
        std::cerr << "Error: Failed to open kernel file " << kernelPath << "\n";
        return -1;
    }
    std::stringstream buffer;
    buffer << kernelFile.rdbuf();
    std::string kernelStr = buffer.str();
    if (kernelStr.empty()) {
        std::cerr << "Error: Kernel file " << kernelPath << " is empty!\n";
        return -1;
    }
    const char* kernel_c_str = kernelStr.c_str();
    size_t kernel_len = kernelStr.length();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_c_str, &kernel_len, &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL build failed (error code " << err << ")\n";
        if (program && device) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "Build log:\n" << log.data() << std::endl;
        }
        return -1;
    }
    // clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ...)

    cl_kernel kernel = clCreateKernel(program, "convolve_fp32", &err);

    // --- 1.4 Memory (Host): Allocate and initialize host data ---
    int width = 1024;
    int height = 1024;
    int k_size = 3;
    size_t img_size_bytes = width * height * sizeof(float);
    size_t kernel_size_bytes = k_size * k_size * sizeof(float);

    std::vector<float> h_input(width * height, 1.0f);
    std::vector<float> h_output(width * height, 0.0f);
    std::vector<float> h_kernel = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };

    // --- 1.5 Memory (Device): Create buffers on the GPU ---
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, img_size_bytes, NULL, &err);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size_bytes, NULL, &err);
    cl_mem d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY, kernel_size_bytes, NULL, &err);

    // --- 1.6 Copy to Device: Write host data to device buffers ---
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, img_size_bytes, h_input.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_kernel, CL_TRUE, 0, kernel_size_bytes, h_kernel.data(), 0, NULL, NULL);

    // --- 1.7 Execute: Set kernel arguments and launch ---
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_kernel);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    clSetKernelArg(kernel, 4, sizeof(int), &height);
    clSetKernelArg(kernel, 5, sizeof(int), &k_size);

    // Define the global work size (one thread per output pixel)
    size_t global_work_size[2] = { (size_t)width, (size_t)height };

    std::cout << "Launching OpenCL Kernel..." << std::endl;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernelEvent);
    clWaitForEvents(1, &kernelEvent);

    // Block until the queue is finished
    //clFinish(queue);
    std::cout << "Kernel finished." << std::endl;
    // Measure time
    cl_ulong timeStart = 0, timeEnd = 0;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, nullptr);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, nullptr);
    double elapsed_us = (timeEnd - timeStart) * 1e-3;
    std::cout << "Kernel execution time: " << elapsed_us << " microseconds\n";

    // --- 1.8 Copy to Host: Read the result buffer back ---
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, img_size_bytes, h_output.data(), 0, NULL, NULL);

    // Step 12. Print first few results
    std::cout << "Output after convolution:\n";
    for (int y = 0; y < height; ++y) {
        if (y < 10 || y >= height - 10) {   // Print only first and last 10 rows
            for (int x = 0; x < width; ++x) {
                if (x < 10 || x >= width - 10) // Print only first and last 10 columns
                {
                    std::cout << h_output[y * width + x] << " ";
                }
                else if (x == 10) {
                    std::cout << "... ";
                }
            }
            std::cout << "\n";
            continue;
        }
        else if (y == 10) {
            std::cout << "... " << std::endl;
        }
    }

    // --- 1.9 Cleanup: Release all OpenCL objects ---
    clReleaseEvent(kernelEvent);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_kernel);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Done. OpenCL resources released." << std::endl;
    // (TODO: Check correctness of h_output)

    return 0;
}

void ConvolutionParallelGPUTests() {
	ConvolutionParallelGPUTest1();
}