#pragma once
#include "hamprod_kernel.h"
#include <curand.h>

/**
 * A simple quaternion-based dense layer that
 * - has N outputs, each is a Quaternion
 * - has M inputs, each is a Quaternion
 * - uses naive (N x M) mat-vec multiply for forward pass
 */
class QuaternionDenseLayer {
private:
    Quaternion* d_W;   // Weights on device, size N*M
    Quaternion* d_b;   // Biases on device, size N
    int N, M;          // Output (N), input (M)
    curandGenerator_t gen;

    void initializeWeights();
public:
    QuaternionDenseLayer(int input_dim, int output_dim);
    ~QuaternionDenseLayer();

    /**
     * Forward pass: out[i] = sum_{j=0..M-1} W[i*M + j] ⊗ in[j] + bias[i].
     */
    void forward(const Quaternion* d_input, Quaternion* d_output);
};
