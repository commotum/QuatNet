#include "hamprod_kernel.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

static inline bool nearlyEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static void cpuHamprod(const Quaternion* A, const Quaternion* B, Quaternion* C, int N) {
    for (int i = 0; i < N; ++i) {
        float a = A[i].w, b = A[i].x, c = A[i].y, d = A[i].z;
        float e = B[i].w, f = B[i].x, g = B[i].y, h = B[i].z;
        C[i].w = a * e - b * f - c * g - d * h;
        C[i].x = a * f + b * e + c * h - d * g;
        C[i].y = a * g - b * h + c * e + d * f;
        C[i].z = a * h + b * g - c * f + d * e;
    }
}

int main() {
    const int N = 32;
    std::vector<Quaternion> h_A(N), h_B(N), h_C(N), h_ref(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = {static_cast<float>(i+1), 0.1f* i, -0.05f*i, 0.2f*i};
        h_B[i] = {0.5f*i, -0.1f*i, 0.3f*i, -0.2f*i};
    }

    Quaternion *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*sizeof(Quaternion));
    cudaMalloc(&d_B, N*sizeof(Quaternion));
    cudaMalloc(&d_C, N*sizeof(Quaternion));

    cudaMemcpy(d_A, h_A.data(), N*sizeof(Quaternion), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N*sizeof(Quaternion), cudaMemcpyHostToDevice);

    launchHamprod(d_A, d_B, d_C, N);

    cudaMemcpy(h_C.data(), d_C, N*sizeof(Quaternion), cudaMemcpyDeviceToHost);

    cpuHamprod(h_A.data(), h_B.data(), h_ref.data(), N);

    for (int i = 0; i < N; ++i) {
        assert(nearlyEqual(h_C[i].w, h_ref[i].w));
        assert(nearlyEqual(h_C[i].x, h_ref[i].x));
        assert(nearlyEqual(h_C[i].y, h_ref[i].y));
        assert(nearlyEqual(h_C[i].z, h_ref[i].z));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("Hamprod kernel test PASS\n");
    return 0;
}
