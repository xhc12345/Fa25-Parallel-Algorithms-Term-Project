// 2. Device (OpenCL C) Kernel
__kernel void convolve_fp32(
    __global const float* input,    // Read-only input image buffer
    __global float* output,         // Write-only output image buffer
    __global const float* k_buffer, // Read-only kernel buffer
    const int width,                // Image width
    const int height,               // Image height
    const int k_size) {             // Kernel size (e.g., 3 for 3x3)
    
    // Get the (x, y) coordinate of the output pixel this thread is responsible for
    int x = get_global_id(0);
    int y = get_global_id(1);

    // (TODO: Add a check to make sure x and y are within bounds
    // if (x >= width || y >= height) return;)

    int k_half = k_size / 2;
    float sum = 0.0f;

    // Perform the convolution (logic is identical to sequential)
    for (int ky = -k_half; ky <= k_half; ++ky) {
        for (int kx = -k_half; kx <= k_half; ++kx) {
            
            int iy = y + ky;
            int ix = x + kx;

            // Simple "clamp-to-edge" padding
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int input_idx = iy * width + ix;
                int kernel_idx = (ky + k_half) * k_size + (kx + k_half);
                sum += input[input_idx] * k_buffer[kernel_idx];
            }
        }
    }
    
    // Write the result to the output buffer
    output[y * width + x] = sum;
}


/**
 * @brief OpenCL kernel for 2D convolution using INT8 (char) data types.
 *
 * This kernel is designed for post-training quantized models.
 * It performs the convolution using 8-bit inputs and kernels,
 * accumulates the result in a 32-bit integer to prevent overflow,
 * and then requantizes the final 32-bit sum back to an 8-bit output.
 *
 * @param input             Read-only input image buffer (int8)
 * @param output            Write-only output image buffer (int8)
 * @param k_buffer          Read-only kernel buffer (int8)
 * @param width             Image width in pixels
 * @param height            Image height in pixels
 * @param k_size            Kernel size (e.g., 3 for 3x3)
 * @param requant_scale     The scaling factor to apply to the 32-bit accumulator. (e.g., input_scale * kernel_scale / output_scale)
 * @param output_zero_point The zero-point of the output tensor.
 */
__kernel void convolve_int8(
    __global const char* input,     // Read-only input image buffer (int8)
    __global char* output,          // Write-only output image buffer (int8)
    __global const char* k_buffer,  // Read-only kernel buffer (int8)
    const int width,                // Image width
    const int height,               // Image height
    const int k_size,               // Kernel size (e.g., 3 for 3x3)
    const float requant_scale,      // Requantization scaling factor
    const int output_zero_point     // Requantization zero point for the output
) {
    
    // Get the (x, y) coordinate of the output pixel
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Bounds check: Ensure the thread is within the output image dimensions
    if (x >= width || y >= height) {
        return;
    }

    int k_half = k_size / 2;
    
    // --- Accumulation ---
    // The accumulator MUST be a 32-bit integer (int) to prevent
    // overflow from the sum of (int8 * int8) products.
    int sum = 0;

    // Perform the convolution
    for (int ky = -k_half; ky <= k_half; ++ky) {
        for (int kx = -k_half; kx <= k_half; ++kx) {
            
            int iy = y + ky; // Input y-coordinate
            int ix = x + kx; // Input x-coordinate

            // Simple "clamp-to-edge" padding
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int input_idx = iy * width + ix;
                int kernel_idx = (ky + k_half) * k_size + (kx + k_half);
                
                // --- Core Quantized Operation ---
                // 1. Cast inputs from char (int8) to int (int32)
                // 2. Perform 32-bit integer multiplication
                // 3. Add to the 32-bit accumulator
                sum += (int)input[input_idx] * (int)k_buffer[kernel_idx];
            }
            // Note: For pixels outside the boundary, 'sum' remains unchanged,
            // effectively adding zero, which is correct for "zero-padding".
        }
    }
    
    // --- Requantization Step ---
    
    // 1. Apply the floating-point scaling factor to the 32-bit sum
    float scaled_sum = (float)sum * requant_scale;
    
    // 2. Round to nearest integer and add the output zero-point
    // (We cast zero_point to float for the addition)
    float requantized_float = round(scaled_sum) + (float)output_zero_point;
    
    // 3. Clamp the value to the int8 range [-128, 127] and convert
    //    'convert_char_sat' converts a float to a char with saturation.
    char final_output = convert_char_sat(requantized_float);
    
    // Write the final requantized int8 result to the output buffer
    output[y * width + x] = final_output;
}