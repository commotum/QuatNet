// src/isokawa_layer.cu
#include "isokawa_layer.h" // Kernel declarations
#include "quat_ops.h"      // For Quaternion struct and its __device__ methods
#include "utils.h"         // For CUDA_CHECK (though typically used in host code launching kernels)

// --- FORWARD PASS KERNELS ---

__global__ void isokawaRotationalForwardKernel(
    const Quaternion* __restrict__ W_q,           // Shape: (N_out, M_in, 4)
    const Quaternion* __restrict__ X_batch,     // Shape: (batch_size, M_in, 4)
    const Quaternion* __restrict__ theta_q,       // Shape: (N_out, 4)
    Quaternion* __restrict__ S_internal_batch,  // Output: (batch_size, N_out, 4)
    int M_in,
    int N_out,
    int batch_size)
{
    int batch_idx = blockIdx.y; // One block per batch sample in y-dimension of grid
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output neuron index (0 to N_out-1)

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }

    Quaternion s_j_accumulator = {0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator for the sum

    const Quaternion* current_X_sample = X_batch + batch_idx * M_in;
    const Quaternion* W_j_row = W_q + j * M_in; // Start of the j-th row of weights

    for (int i = 0; i < M_in; ++i) {
        Quaternion w_ji = W_j_row[i];      // Weight from input i to output j
        Quaternion x_i = current_X_sample[i]; // Input i (x_i.w should be 0 for pure input)

        float w_norm = w_ji.norm(); // Uses stabilized norm from quat_ops.h
        Quaternion u_ji;
        if (w_norm < 1e-6f) { // Avoid division by zero or issues with very small norms
            u_ji = {0.0f, 0.0f, 0.0f, 0.0f}; // Or handle as w_ji if norm is tiny (no rotation)
        } else {
            u_ji = w_ji.scale(1.0f / w_norm);
        }

        // Rotational operation: u_ji * x_i * u_ji.conjugate()
        // Since x_i.w is ~0, the result will also have w ~0.
        s_j_accumulator = s_j_accumulator + (u_ji * x_i * u_ji.conjugate());
    }

    // Subtract threshold (theta_q[j].w should be 0 for pure threshold)
    s_j_accumulator = s_j_accumulator - theta_q[j];

    S_internal_batch[batch_idx * N_out + j] = s_j_accumulator; // s_j will be pure
}

__device__ float device_sigmoid(float val) {
    return 1.0f / (1.0f + expf(-val));
}

__global__ void isokawaActivationKernel(
    const Quaternion* __restrict__ S_internal_batch, // Input: (batch_size, N_out, 4)
    Quaternion* __restrict__ Y_batch,                // Output: (batch_size, N_out, 4)
    int N_out,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output neuron index

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }

    int global_idx = batch_idx * N_out + j;
    Quaternion s_j = S_internal_batch[global_idx];

    // s_j.w should be effectively zero from previous calculations.
    // Apply sigmoid to imaginary components, output is pure quaternion.
    Y_batch[global_idx] = {
        0.0f,
        device_sigmoid(s_j.x),
        device_sigmoid(s_j.y),
        device_sigmoid(s_j.z)
    };
}

// --- BACKWARD PASS KERNEL DEFINITIONS (STUBS for now) ---

__global__ void isokawaActivationBackwardKernel(
    const Quaternion* __restrict__ grad_Y_batch,
    const Quaternion* __restrict__ S_internal_batch,
    Quaternion* __restrict__ grad_S_internal_batch,
    int N_out,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }
    // TODO: Implement dL/dS = dL/dY * f'(S)
    // f'(s_comp) = sigmoid(s_comp) * (1 - sigmoid(s_comp))
    // For now, just pass through or zero out grad_S_internal_batch[global_idx]
    int global_idx = batch_idx * N_out + j;
    Quaternion s_j = S_internal_batch[global_idx];
    Quaternion grad_y_j = grad_Y_batch[global_idx];

    float sig_sx = device_sigmoid(s_j.x);
    float sig_sy = device_sigmoid(s_j.y);
    float sig_sz = device_sigmoid(s_j.z);

    grad_S_internal_batch[global_idx] = {
        0.0f, // Gradient for w component is 0 as it's a pure quaternion operation
        grad_y_j.x * sig_sx * (1.0f - sig_sx),
        grad_y_j.y * sig_sy * (1.0f - sig_sy),
        grad_y_j.z * sig_sz * (1.0f - sig_sz)
    };
}

__global__ void isokawaRotationalBackwardKernel(
    const Quaternion* __restrict__ grad_S_internal_batch,
    const Quaternion* __restrict__ X_batch,
    const Quaternion* __restrict__ W_q,
    Quaternion* __restrict__ grad_X_batch,
    Quaternion* __restrict__ grad_W_q,
    Quaternion* __restrict__ grad_theta_q,
    int M_in,
    int N_out,
    int batch_size)
{
    // This is the most complex kernel to implement.
    // It requires careful derivation of quaternion calculus.
    // For now, it's a stub. Gradients should be accumulated if multiple threads contribute.
    // Example for grad_theta_q (simplest part):
    int batch_idx = blockIdx.y; // Iterate over batches if one block computes all N_out grads
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output neuron index

    if (j >= N_out) { // Each thread computes one grad_theta_q[j] element
        return;
    }

    // dL/d(theta_j) = sum_batch (dL/dS_j * dS_j/d(theta_j)) = sum_batch (dL/dS_j * (-1))
    // Needs atomicAdd if multiple blocks work on same N_out, or reduction.
    // If one block per N_out, and threads in block sum over batch_size:
    if (threadIdx.x == 0 && blockIdx.x < N_out) { // One thread per block for one theta_j
        Quaternion grad_theta_sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
            grad_theta_sum = grad_theta_sum - grad_S_internal_batch[b_idx * N_out + blockIdx.x];
        }
        grad_theta_q[blockIdx.x] = grad_theta_sum;
    }

    // TODO: Implement grad_X_batch and grad_W_q
    // grad_X_batch will be zeroed out by default from PyTorch if not written to.
    // grad_W_q will be zeroed out by default.
}