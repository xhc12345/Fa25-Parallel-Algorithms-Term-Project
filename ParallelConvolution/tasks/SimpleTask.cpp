#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

void runSimpleOpenCLTask(int N) {
    // Step 1. Get platform
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Step 2. Get device (first GPU)
    cl_uint numDevices = 0;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    // Step 3. Create context and queue (profiling enabled)
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue_properties queueProps[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queueProps, &err);

    // Step 4. Load kernel source safely
    std::string kernelPath = "kernels/add.cl";
    std::ifstream kernelFile(kernelPath);
    if (!kernelFile.is_open()) {
        std::cerr << "Error: Failed to open kernel file " << kernelPath << "\n";
        return;
    }
    std::stringstream buffer;
    buffer << kernelFile.rdbuf();
    std::string sourceStr = buffer.str();
    if (sourceStr.empty()) {
        std::cerr << "Error: Kernel file " << kernelPath << " is empty!\n";
        return;
    }
    const char* source = sourceStr.c_str();

    // Step 5. Build program
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL build failed (error code " << err << ")\n";
        if (program && device) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "Build log:\n" << log.data() << std::endl;
        }
        return;
    }

    // Step 6. Create kernel
    cl_kernel kernel = clCreateKernel(program, "add", &err);

    // Step 7. Prepare data
    std::vector<float> a(N), b(N), result(N);
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(10 * i);
    }

    // Step 8. Create buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, a.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, b.data(), &err);
    cl_mem bufR = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, nullptr, &err);

    // Step 9. Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufR);

    // Step 10. Run kernel and measure execution time
    size_t globalSize = N;
    cl_event kernelEvent;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, &kernelEvent);
    clWaitForEvents(1, &kernelEvent);

    // Measure time
    cl_ulong timeStart = 0, timeEnd = 0;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, nullptr);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, nullptr);
    double elapsed_us = (timeEnd - timeStart) * 1e-3;
    std::cout << "Kernel execution time: " << elapsed_us << " microseconds\n";

    // Step 11. Read back
    clEnqueueReadBuffer(queue, bufR, CL_TRUE, 0, sizeof(float) * N, result.data(), 0, nullptr, nullptr);

    // Step 12. Print first few results
    std::cout << "First 10 results: ";
    for (int i = 0; i < std::min(N, 10); i++)
        std::cout << result[i] << " ";
    std::cout << (N > 10 ? "... " : "") << "\n";

    // Step 13. Cleanup
    clReleaseEvent(kernelEvent);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufR);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


void SimpleTaskTest() {
    int N;

    N = 100'000;
    std::cout << "Running simple OpenCL task, N=" << N << std::endl;
    runSimpleOpenCLTask(N);
    std::cout << std::endl;

    N = 1'000'000;
    std::cout << "Running simple OpenCL task, N=" << N << std::endl;
    runSimpleOpenCLTask(N);
    std::cout << std::endl;

    N = 10'000'000;
    std::cout << "Running simple OpenCL task, N=" << N << std::endl;
    runSimpleOpenCLTask(N);
    std::cout << std::endl;
}
