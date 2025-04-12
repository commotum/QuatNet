// quatnet_layer.cpp
#include "hamprod_kernel.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>

class QuatnetDenseLayer {
private:
    Quaternion* d_W; // Device weights (N x M)
    Quaternion* d_b; // Device biases (N)
    int N, M;        // Output and input dimensions
    curandGenerator_t gen;

    void initializeWeights() {
        float* temp;
        cudaMalloc(&temp, N * M * 4 * sizeof(float));
        curandGenerateUniform(gen, temp, N * M * 4);
        cudaMemcpy(d_W, temp, N * M * sizeof(Quaternion), cudaMemcpyDeviceToDevice);
        cudaFree(temp);
        cudaMalloc(&temp, N * 4 * sizeof(float));
        curandGenerateUniform(gen, temp, N * 4);
        cudaMemcpy(d_b, temp, N * sizeof(Quaternion), cudaMemcpyDeviceToDevice);
        cudaFree(temp);
    }

public:
    QuatnetDenseLayer(int input_dim, int output_dim) : M(input_dim), N(output_dim) {
        cudaError_t err;
        err = cudaMalloc(&d_W, N * M * sizeof(Quaternion));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_W");
        err = cudaMalloc(&d_b, N * sizeof(Quaternion));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_b");
        
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        initializeWeights();
    }

    ~QuatnetDenseLayer() {
        cudaFree(d_W);
        cudaFree(d_b);
        curandDestroyGenerator(gen);
    }

    void forward(const Quaternion* d_input, Quaternion* d_output) {
        launchQuatMatVecMul(d_W, d_input, d_output, N, M);
        launchAddBias(d_output, d_b, N);
    }
};