// quatnet_layer.cpp
#include "hamprod_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

class QuatnetDenseLayer {
private:
    Quaternion* d_W; // Device weights (N x M)
    Quaternion* d_b; // Device biases (N)
    int N, M;        // Output and input dimensions
public:
    QuatnetDenseLayer(int input_dim, int output_dim) : M(input_dim), N(output_dim) {
        // Allocate weights and biases on GPU
        cudaMalloc(&d_W, N * M * sizeof(Quaternion));
        cudaMalloc(&d_b, N * sizeof(Quaternion));
        // Initialize to zero (use cuRAND for random in practice)
        cudaMemset(d_W, 0, N * M * sizeof(Quaternion));
        cudaMemset(d_b, 0, N * sizeof(Quaternion));
    }

    ~QuatnetDenseLayer() {
        cudaFree(d_W);
        cudaFree(d_b);
    }

    void forward(const Quaternion* d_input, Quaternion* d_output) {
        // Compute W * input (simplified elementwise; matrix version later)
        launchHamprod(d_W, d_input, d_output, N * M);
        // Biases omitted for simplicity (assumed zero)
    }
};