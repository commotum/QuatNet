// src/quat_ops.h
#pragma once
#include <cuda_runtime.h> // For __device__ and float
#include <cmath>          // For std::sqrt, std::fabs, std::exp, std::log, std::cos on host

// Define M_PI if not already defined (cmath should provide it)
#ifndef M_PI
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
#ifdef __CUDA_ARCH__
        return sqrtf(n_sq + 1e-12f);
#else
        return std::sqrt(n_sq + 1e-12f);
#endif
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

// Atomic addition for Quaternion values (device only)
__device__ inline void atomicAddQuaternion(Quaternion* addr, const Quaternion& val) {
    atomicAdd(&(addr->w), val.w);
    atomicAdd(&(addr->x), val.x);
    atomicAdd(&(addr->y), val.y);
    atomicAdd(&(addr->z), val.z);
}

// Device-side sigmoid for individual components
__device__ inline float device_component_sigmoid(float val) {
    return 1.0f / (1.0f + expf(-val));
}

// Device-side tanh for individual components
__device__ inline float device_component_tanh(float val) {
    return tanhf(val);
}
