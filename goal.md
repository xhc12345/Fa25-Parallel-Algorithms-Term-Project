# Project Outline: Parallel CPU/GPU Convolution (OpenCL)

## 1. Introduction

* **Problem Definition:** What is 2D Convolution?
    * Explain the operation: A kernel slides over an input, performing element-wise multiplication and summation.
    * Mention its use in CNNs.
* **Project Goal:**
    * To implement and compare the performance of 2D convolution using three different methods:
        1.  **Sequential (CPU):** A single-threaded C++ baseline.
        2.  **Parallel (CPU):** A multi-threaded C++ implementation **using OpenMP**.
        3.  **Parallel (GPU):** A massively parallel GPU implementation **using OpenCL**.
* **Hypothesis:**
    * (Same as before) Sequential will be fast for small inputs. OpenMP will scale with CPU cores. OpenCL will be *much* faster for large inputs, despite its initial setup and memory-copy overhead.

## 2. Implementations

* **2.1. Sequential (Baseline)**
    * **Language:** C++
    * **Algorithm:** Simple, naive nested loops. This is your "correctness" check.

* **2.2. Parallel CPU (OpenMP)**
    * **Technology:** **OpenMP**
    * **Algorithm:**
        * Take the sequential C++ code.
        * Add `#pragma omp parallel for` to the outermost loop (over the output rows).
        * This is the standard, high-level approach for multi-core CPU parallelism.

* **2.3. Parallel GPU (OpenCL)**
    * **Technology:** **OpenCL**
    * **Algorithm:** This has two parts: a C++ "host" program and an "OpenCL C" kernel.
        1.  **Host (C++) Code:**
            * **Setup:** Find the platform (e.g., AMD, Intel) and device (your GPU). Create an OpenCL `context` and a `command_queue`.
            * **Kernel:** Write the kernel code (see below) *as a text string* inside your C++ program.
            * **Build:** Create a `program` from that string (`clCreateProgramWithSource`) and `build` it (`clBuildProgram`). This compiles your kernel *at runtime*.
            * **Memory (Host):** Allocate and initialize your input image and kernel on the CPU.
            * **Memory (Device):** Create buffers on the GPU (`clCreateBuffer`) for the input, kernel, and output.
            * **Copy to Device:** Write your input/kernel data from the host to the device buffers (`clEnqueueWriteBuffer`).
            * **Execute:** Set the kernel's arguments (`clSetKernelArg`) and launch the kernel (`clEnqueueNDRangeKernel`).
            * **Copy to Host:** Read the result buffer from the device back to the host (`clEnqueueReadBuffer`).
            * **Cleanup:** Release all the OpenCL objects (`clRelease...`).
        2.  **Device (OpenCL C) Kernel (`__kernel` function):**
            * This is the code that runs on the GPU. The logic is *identical* to the CUDA plan.
            * `__kernel void convolve(...)`
            * Use `get_global_id(0)` and `get_global_id(1)` to get the (x, y) coordinate of the output pixel this thread is responsible for.
            * Each thread performs the small, inner loops (over the kernel size) to calculate the value for its single output pixel.

## 3. Experimental Setup

* **Hardware:**
    * **CPU:** Model and number of cores (e.g., "AMD Ryzen 7 5800X, 8 Cores, 16 Threads").
    * **GPU:** Model (e.g., "AMD Radeon RX 6800" or "Intel Iris Xe Graphics").
* **Software:**
    * Compiler (e.g., `g++` with `-O3` and `-fopenmp`).
    * OpenCL SDK (e.g., from AMD, Intel, or the Khronos registry).
* **Test Data & Metrics:**
    * (Same as before) Test with various kernel sizes (3x3, 5x5) and a wide range of input sizes (256x256 up to 8192x8192).
    * Measure wall clock time (ms) and calculate speedup (`Time_Sequential / Time_Parallel`).

## 4. Analysis & Expected Results

* (Same as before)
* **Generate Plots:** Time vs. Input Size (Seq, OMP, OCL) and Speedup vs. Input Size (OMP, OCL).
* **Discuss:** Compare the three methods. At what size does the OpenCL overhead (setup + memory copy) become worth it? How does the OpenMP speedup compare to your number of CPU cores?

## 5. Conclusion & (Optional) Future Work

* (Same as before)
* Summarize your findings about the performance and trade-offs of the three different parallel programming models.