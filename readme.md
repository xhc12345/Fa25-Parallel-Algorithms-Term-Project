# Parallel Convolution

This project benchmarks various parallization strategy for convolution.

Please read the paper in the base folder for more information.

## My Hardware Setup

CPU: AMD Ryzen 9 6900HS, Zen 3+, 8 cores 16 threads
RAM: 2x8GB DDR4
GPU0: AMD Radeon RX 6700S (Discrete, Navi 23, RDNA2, 28CU)
GPU1: AMD Radeon 680M (Integrated, Radeon 680M, RDNA2, 12CU)

## Code Structure

The project consists of two distinct parts:

1. C++ Benchmark of various parallel code in `ParallelConvolution` folder

2. Web application serving CNN MNIST models in `DrawApp` folder

Please go to each folder for further instructions.

## Hardware/Software Requirements

For a worry-free compilation process, you MUST have:

1. Windows Operating System, Windows 10 or newer. (Sorry)

2. OpenMP-enabled platform.

3. Visual Studio IDE, with support for C++14 compilation.

4. OpenCL-enabled graphic drivers.

5. OpenCL SDK installed and verified in PATH.

6. Multi-core x86-64 CPU with support for AVX2 instruction set.

7. 4GB+ RAM (The benchmark consumes a lot of RAM)

8. Python 3.12 with capability to initialize virtual environments via `venv`.

## Data

If you don't feel like going through the previous requirements, I have all the data collected in the `data` folder for you. The figures genreated using them are in the `figures` folder, which in turn is used by the paper.