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
        float a = A[idx].w, b = A[idx].x, c = A[idx].y, d = A[idx].z;
        float e = B[idx].w, f = B[idx].x, g = B[idx].y, h = B[idx].z;
        float rw = a * e - b * f - c * g - d * h;
        float rx = a * f + b * e + c * h - d * g;
        float ry = a * g - b * h + c * e + d * f;
        float rz = a * h + b * g - c * f + d * e;
        C[idx].w = rw;
        C[idx].x = rx;
        C[idx].y = ry;
        C[idx].z = rz;
    }
}

void launchHamprod(const Quaternion* A, const Quaternion* B,
                   Quaternion* C, int N) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    hamprodKernel<<<gridDim, blockDim>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

__global__ void quatMatVecMul(const Quaternion* W, const Quaternion* input,
                              Quaternion* output, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Quaternion sum = {0, 0, 0, 0};
        for (int j = 0; j < M; ++j) {
            Quaternion w = W[i * M + j];
            Quaternion in = input[j];
            Quaternion prod;
            prod.w = w.w * in.w - w.x * in.x - w.y * in.y - w.z * in.z;
            prod.x = w.w * in.x + w.x * in.w + w.y * in.z - w.z * in.y;
            prod.y = w.w * in.y - w.x * in.z + w.y * in.w + w.z * in.x;
            prod.z = w.w * in.z + w.x * in.y - w.y * in.x + w.z * in.w;
            sum.w += prod.w;
            sum.x += prod.x;
            sum.y += prod.y;
            sum.z += prod.z;
        }
        output[i] = sum;
    }
}

void launchQuatMatVecMul(const Quaternion* W, const Quaternion* input,
                         Quaternion* output, int N, int M) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    quatMatVecMul<<<gridDim, blockDim>>>(W, input, output, N, M);
    cudaDeviceSynchronize();
}

__global__ void addBiasKernel(Quaternion* output, const Quaternion* bias, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i].w += bias[i].w;
        output[i].x += bias[i].x;
        output[i].y += bias[i].y;
        output[i].z += bias[i].z;
    }
}

void launchAddBias(Quaternion* output, const Quaternion* bias, int N) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    addBiasKernel<<<gridDim, blockDim>>>(output, bias, N);
    cudaDeviceSynchronize();
}