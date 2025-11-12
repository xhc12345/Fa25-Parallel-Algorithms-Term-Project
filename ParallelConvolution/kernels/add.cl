__kernel void add(__global const float* a, __global const float* b, __global float* result) {
    int i = get_global_id(0);
    result[i] = a[i] + b[i];
}