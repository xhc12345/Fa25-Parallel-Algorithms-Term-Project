// ParallelConvolution.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <CL/cl.h>
#include <iostream>
#include "tasks/SimpleTask.h"
#include <vector>
#include "tasks/convolution_seq.h"
#include "tasks/convolution_omp.h"
#include "tasks/convolution_ocl.h"

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
        std::cout << "\n--- Platform " << i << ": " << platform_name << " ---" << std::endl;

        // 4. Get the number of devices on this platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            std::cout << "  No devices found on this platform." << std::endl;
            continue;
        }

        std::cout << "  Found " << num_devices << " device(s)." << std::endl;

        // 5. Get the device IDs
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL);

        // 6. Iterate over each device and print its name
        for (cl_uint j = 0; j < num_devices; ++j) {
            char device_name[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, device_name, NULL);
            std::cout << "    Device " << j << ": " << device_name << std::endl;
        }
    }

    std::cout << "\nOpenCL check complete." << std::endl;
}




int main() {
    OpenCLTest();

    //SimpleTaskTest();

    ConvolutionSequentialCPUTests();

    ConvolutionParallelCPUTests();

    ConvolutionParallelGPUTests();
    return 0;
}
