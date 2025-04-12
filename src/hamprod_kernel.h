// hamprod_kernel.h
#pragma once

struct Quaternion {
    float w, x, y, z;
};

void launchHamprod(const Quaternion* A, const Quaternion* B,
                   Quaternion* C, int N);