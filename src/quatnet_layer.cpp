#include "quatnet_layer.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

/**
 * Here, we do a simple normal-based init for each quaternion component.
 * Each weight is (w, x, y, z), each dimension ~ N(0, sigma^2).
 * Then the overall magnitude of W is \chi_4-distributed, with expectation
 * E(|W|^2)=4*sigma^2, as shown in the derivation.
 */
void QuaternionDenseLayer::initializeWeights() {
    // For a typical "Xavier"-style variance, you might do:
    //   sigma^2 = 1 / fan_in
    // or some variant. You can adjust the formula as needed.
    float fan_in = static_cast<float>(M); // each input is a Quaternion, but let's keep it simple
    float fan_out = static_cast<float>(N);

    // Example: "Xavier uniform" for quaternions is not standardized in literature,
    // but let's just pick something analogous:
    //   sigma^2 = 2 / (fan_in + fan_out)
    // or you can do a simple 1 / fan_in:
    float sigma2 = 2.0f / (fan_in + fan_out);
    float sigma  = sqrtf(sigma2);

    // We have N*M quaternions (4 floats each) + N quaternions for biases (4 floats each).
    size_t numW = static_cast<size_t>(N)*M;  // # of quaternions in W
    size_t numB = static_cast<size_t>(N);    // # of quaternions in bias
    size_t totalQuaternions = numW + numB;
    size_t totalFloats = 4 * totalQuaternions; // 4 floats per quaternion

    // We'll generate normal random floats (mean=0, stddev=sigma) in device memory
    float* d_randFloats = nullptr;
    cudaMalloc(&d_randFloats, totalFloats * sizeof(float));

    // Generate normal distribution on GPU:
    curandGenerateNormal(gen, d_randFloats, totalFloats, 0.0f, sigma);

    // Copy the first 4*numW floats into d_W, the next 4*N floats into d_b.
    // Because d_W and d_b are arrays of Quaternion, each Quaternion is 4 floats.
    // So we can do a single cudaMemcpy of 4*numW floats into d_W, etc.
    cudaMemcpy(d_W, d_randFloats, numW * sizeof(Quaternion), cudaMemcpyDeviceToDevice);

    // Bias floats start after the W floats
    size_t offset = numW * 4;
    cudaMemcpy(d_b, d_randFloats + offset, numB * sizeof(Quaternion), cudaMemcpyDeviceToDevice);

    cudaFree(d_randFloats);
}

QuaternionDenseLayer::QuaternionDenseLayer(int input_dim, int output_dim)
    : M(input_dim), N(output_dim)
{
    // Allocate device memory for W (N*M) and biases (N).
    cudaError_t err = cudaMalloc(&d_W, N * M * sizeof(Quaternion));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_W");

    err = cudaMalloc(&d_b, N * sizeof(Quaternion));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_b");

    // Create RNG generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Set a fixed seed or something else:
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Initialize weights/bias
    initializeWeights();
}

QuaternionDenseLayer::~QuaternionDenseLayer() {
    cudaFree(d_W);
    cudaFree(d_b);
    curandDestroyGenerator(gen);
}

void QuaternionDenseLayer::forward(const Quaternion* d_input, Quaternion* d_output) {
    // 1) Multiply W x input
    //    Each of N threads accumulates over M quaternions
    launchQuatMatVecMul(d_W, d_input, d_output, N, M);

    // 2) Add bias
    launchAddBias(d_output, d_b, N);
}
