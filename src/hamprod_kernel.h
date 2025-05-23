// src/hamprod_kernel.h
#pragma once
#include <cuda_runtime.h> // For __device__ and float
#include <cmath>          // For sqrtf, fabsf

struct Quaternion {
    float w, x, y, z;

#ifdef __CUDA_ARCH__ // Only compile for CUDA device code
    // Default constructor
    __device__ Quaternion(float _w=0.f, float _x=0.f, float _y=0.f, float _z=0.f) : w(_w), x(_x), y(_y), z(_z) {}

    // Conjugate
    __device__ Quaternion conjugate() const {
        return {w, -x, -y, -z};
    }

    // Norm squared
    __device__ float norm_sq() const {
        return w * w + x * x + y * y + z * z;
    }

    // Norm
    __device__ float norm() const {
        return sqrtf(norm_sq());
    }

    // Scale
    __device__ Quaternion scale(float s) const {
        return {w * s, x * s, y * s, z * s};
    }

    // Hamilton Product (operator overloading for convenience)
    __device__ Quaternion operator*(const Quaternion& other) const {
        return {
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        };
    }

    // Addition (operator overloading)
    __device__ Quaternion operator+(const Quaternion& other) const {
        return {w + other.w, x + other.x, y + other.y, z + other.z};
    }
    
    // Subtraction (operator overloading)
    __device__ Quaternion operator-(const Quaternion& other) const {
        return {w - other.w, x - other.x, y - other.y, z - other.z};
    }
#endif
};

// --- Existing Global Kernel Launchers (if still needed for other parts) ---
// Element-wise Hamilton Product C = A * B
void launchHamprod(const Quaternion* A, const Quaternion* B,
                   Quaternion* C, int N);

// Matrix-Vector W*x (QuaternionDenseLayer style)
void launchQuatMatVecMul(const Quaternion* W, const Quaternion* input,
                         Quaternion* output, int N_out, int M_in);

// Add bias
void launchAddBias(Quaternion* output, const Quaternion* bias, int N_out);

// --- New Device Functions (Prototypes - definitions will be in .cu) ---
// These are illustrative; we'll use operator overloading within the struct for device code.
// __device__ Quaternion conjugateDevice(const Quaternion& q);
// __device__ float normDevice(const Quaternion& q);
// __device__ Quaternion hamiltonProductDevice(const Quaternion& q1, const Quaternion& q2);
// __device__ Quaternion scaleDevice(const Quaternion& q, float s);