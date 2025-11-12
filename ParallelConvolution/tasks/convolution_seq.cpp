#include <iostream>
#include <vector>
#include <chrono>   // 1. Include for C++ timing
#include <iomanip>  // 2. Include for output formatting

void sequential_convolution(const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<float>& kernel,
    int width, int height, int k_size) {

    int k_half = k_size / 2;

    // Iterate over each output pixel (y, x)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            float sum = 0.0f;

            // Iterate over the kernel (ky, kx)
            for (int ky = -k_half; ky <= k_half; ++ky) {
                for (int kx = -k_half; kx <= k_half; ++kx) {

                    int iy = y + ky;
                    int ix = x + kx;

                    // Simple "clamp-to-edge" padding
                    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        // (y, x) in 2D -> (y * width + x) in 1D
                        int input_idx = iy * width + ix;
                        // (ky, kx) in 2D -> ((ky + k_half) * k_size + (kx + k_half)) in 1D
                        int kernel_idx = (ky + k_half) * k_size + (kx + k_half);
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

int ConvolutionSeqTest1() {
    const int width = 5;
    const int height = 5;
    const int k_size = 3;
    std::vector<float> input = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    std::vector<float> kernel = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };
    std::vector<float> output(width * height, 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    sequential_convolution(input, output, kernel, width, height, k_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Sequential execution time (Test 1): " << duration.count() << " microseconds." << std::endl;
    std::cout << "Output after convolution:" << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << output[y * width + x] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}

int ConvolutionSeqTest2() {
    // 1. Setup your test data
    int width = 1024;
    int height = 1024;
    int k_size = 3; // e.g., 3x3

    std::vector<float> input(width * height, 1.0f); // Example: all 1s
    std::vector<float> output(width * height, 0.0f);

    // Example: 3x3 identity kernel (does nothing)
    std::vector<float> kernel = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };

    std::cout << "Running Sequential Convolution..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    sequential_convolution(input, output, kernel, width, height, k_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Sequential execution time (Test 1): " << duration.count() << " microseconds." << std::endl;

    std::cout << "Output after convolution:" << std::endl;
    for (int y = 0; y < height; ++y) {
        if (y < 10 || y >= height - 10) {   // Print only first and last 10 rows
            for (int x = 0; x < width; ++x) {
                if (x < 10 || x >= width - 10) // Print only first and last 10 columns
                {
                    std::cout << output[y * width + x] << " ";
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
    // 4. (TODO) Add correctness check against a known good result
    return 0;
}

void ConvolutionSequentialCPUTests() {
    std::cout << "---------------------------------" << std::endl; \
    ConvolutionSeqTest1();
	std::cout << "---------------------------------" << std::endl;\
    ConvolutionSeqTest2();
    std::cout << "---------------------------------" << std::endl; \
}