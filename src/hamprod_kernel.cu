// hamprod_kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Quaternion {
    float w, x, y, z;
};

__global__ void hamprodKernel(const Quaternion* __restrict__ A,
                              const Quaternion* __restrict__ B,
                              Quaternion* __restrict__ C,
                              int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Load quaternions
        float a = A[idx].w, b = A[idx].x, c = A[idx].y, d = A[idx].z;
        float e = B[idx].w, f = B[idx].x, g = B[idx].y, h = B[idx].z;
        // Compute Hamilton product
        float rw = a * e - b * f - c * g - d * h;
        float rx = a * f + b * e + c * h - d * g;
        float ry = a * g - b * h + c * e + d * f;
        float rz = a * h + b * g - c * f + d * e;
        // Store result
        C[idx].w = rw;
        C[idx].x = rx;
        C[idx].y = ry;
        C[idx].z = rz;
    }
}

// Host function to launch kernel
void launchHamprod(const Quaternion* A, const Quaternion* B,
                   Quaternion* C, int N) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    hamprodKernel<<<gridDim, blockDim>>>(A, B, C, N);
    cudaDeviceSynchronize(); // Wait for kernel to finish
}