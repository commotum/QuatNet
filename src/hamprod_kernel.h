// hamprod_kernel.h
#pragma once

struct Quaternion {
    float w, x, y, z;
};

void launchHamprod(const Quaternion* A, const Quaternion* B,
                   Quaternion* C, int N);

void launchQuatMatVecMul(const Quaternion* W, const Quaternion* input,
                         Quaternion* output, int N, int M);

void launchAddBias(Quaternion* output, const Quaternion* bias, int N);