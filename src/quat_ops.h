// src/quat_ops.h
#pragma once
#include <cuda_runtime.h> // For __device__ and float
#include <cmath>          // For sqrtf, fabsf, expf, logf, cosf

#ifndef M_PI // Ensure M_PI is defined
#define M_PI 3.14159265358979323846
#endif

struct Quaternion {
    float w, x, y, z;

    __host__ __device__ Quaternion(float _w = 0.f, float _x = 0.f, float _y = 0.f, float _z = 0.f) 
        : w(_w), x(_x), y(_y), z(_z) {}

    __host__ __device__ Quaternion conjugate() const {
        return {w, -x, -y, -z};
    }

    __host__ __device__ float norm_sq() const {
        return w * w + x * x + y * y + z * z;
    }

    __host__ __device__ float norm() const {
        float n_sq = norm_sq();
        return sqrtf(n_sq + 1e-12f);
    }

    __host__ __device__ Quaternion scale(float s) const {
        return {w * s, x * s, y * s, z * s};
    }

    __host__ __device__ Quaternion operator*(const Quaternion& other) const {
        return {
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        };
    }

    __host__ __device__ Quaternion operator+(const Quaternion& other) const {
        return {w + other.w, x + other.x, y + other.y, z + other.z};
    }
    
    __host__ __device__ Quaternion operator-(const Quaternion& other) const {
        return {w - other.w, x - other.x, y - other.y, z - other.z};
    }
};

// __global__ kernel for element-wise Hamilton product (if needed standalone)
// This was previously in hamprod_kernel.cu
// void launchElementwiseHamprod(const Quaternion* A, const Quaternion* B, Quaternion* C, int N);