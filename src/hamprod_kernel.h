// hamprod_kernel.h
#pragma once


#include "quat_ops.h"

void launchHamprod(const Quaternion* A, const Quaternion* B,
                   Quaternion* C, int N);

void launchQuatMatVecMul(const Quaternion* W, const Quaternion* input,
                         Quaternion* output, int N, int M);

void launchAddBias(Quaternion* output, const Quaternion* bias, int N);
