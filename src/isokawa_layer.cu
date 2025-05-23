// src/isokawa_layer.cu
#include "isokawa_layer.h" // Kernel declarations and new host launcher declarations
#include "quat_ops.h"      // For Quaternion struct and its __device__ methods
#include "utils.h"         // For CUDA_CHECK

// --- KERNEL DEFINITIONS (as provided before) ---

__global__ void isokawaRotationalForwardKernel(
    const Quaternion* __restrict__ W_q,
    const Quaternion* __restrict__ X_batch,
    const Quaternion* __restrict__ theta_q,
    Quaternion* __restrict__ S_internal_batch,
    int M_in,
    int N_out,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }
    Quaternion s_j_accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    const Quaternion* current_X_sample = X_batch + batch_idx * M_in;
    const Quaternion* W_j_row = W_q + j * M_in;

    for (int i = 0; i < M_in; ++i) {
        Quaternion w_ji = W_j_row[i];
        Quaternion x_i = current_X_sample[i];
        float w_norm_val = w_ji.norm();
        Quaternion u_ji;
        if (w_norm_val < 1e-6f) {
            u_ji = {0.0f, 0.0f, 0.0f, 0.0f};
        } else {
            u_ji = w_ji.scale(1.0f / w_norm_val);
        }
        s_j_accumulator = s_j_accumulator + (u_ji * x_i * u_ji.conjugate());
    }
    s_j_accumulator = s_j_accumulator - theta_q[j];
    S_internal_batch[batch_idx * N_out + j] = s_j_accumulator;
}

__device__ float kernel_sigmoid(float val) { // Ensure this is defined or included if used
    return 1.0f / (1.0f + expf(-val));
}

__global__ void isokawaActivationKernel(
    const Quaternion* __restrict__ S_internal_batch,
    Quaternion* __restrict__ Y_batch,
    int N_out,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }
    int global_idx = batch_idx * N_out + j;
    Quaternion s_j = S_internal_batch[global_idx];
    Y_batch[global_idx] = {
        0.0f,
        kernel_sigmoid(s_j.x),
        kernel_sigmoid(s_j.y),
        kernel_sigmoid(s_j.z)
    };
}

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
    int global_idx = batch_idx * N_out + j;
    Quaternion s_j = S_internal_batch[global_idx];
    Quaternion grad_y_j = grad_Y_batch[global_idx];

    float sig_sx = kernel_sigmoid(s_j.x);
    float sig_sy = kernel_sigmoid(s_j.y);
    float sig_sz = kernel_sigmoid(s_j.z);

    grad_S_internal_batch[global_idx] = {
        0.0f,
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
    int j = blockIdx.x; // Output neuron index, assuming gridDim.x = N_out

    if (j >= N_out) {
        return;
    }

    if (threadIdx.x == 0) { // One thread per block for theta_j gradient
        Quaternion grad_theta_sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
            grad_theta_sum = grad_theta_sum - grad_S_internal_batch[b_idx * N_out + j];
        }
        grad_theta_q[j] = grad_theta_sum;
    }
    // TODO: Implement full backward logic for grad_X_batch and grad_W_q.
    // This will likely involve more complex indexing and possibly atomic operations
    // if multiple threads/blocks contribute to the same gradient elements.
    // For grad_W_q[j * M_in + i] and grad_X_batch[b_idx * M_in + i].
}

// --- C++ HOST FUNCTION IMPLEMENTATIONS ---

void launch_isokawa_forward_kernels(
    const Quaternion* d_W_ptr,
    const Quaternion* d_X_ptr,
    const Quaternion* d_theta_ptr,
    Quaternion* d_S_internal_ptr,
    Quaternion* d_Y_ptr,
    int M_in,
    int N_out,
    int batch_size,
    cudaStream_t stream)
{
    dim3 block_dim(256); // Example block size, can be tuned
    dim3 grid_dim;
    grid_dim.x = (N_out + block_dim.x - 1) / block_dim.x;
    grid_dim.y = batch_size;
    grid_dim.z = 1;

    isokawaRotationalForwardKernel<<<grid_dim, block_dim, 0, stream>>>(
        d_W_ptr, d_X_ptr, d_theta_ptr, d_S_internal_ptr, M_in, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch

    isokawaActivationKernel<<<grid_dim, block_dim, 0, stream>>>(
        d_S_internal_ptr, d_Y_ptr, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_isokawa_backward_kernels(
    const Quaternion* d_grad_Y_ptr,
    const Quaternion* d_X_saved_ptr,
    const Quaternion* d_W_saved_ptr,
    const Quaternion* d_S_internal_saved_ptr,
    Quaternion* d_grad_S_internal_ptr,
    Quaternion* d_grad_X_ptr,
    Quaternion* d_grad_W_ptr,
    Quaternion* d_grad_theta_ptr,
    int M_in,
    int N_out,
    int batch_size,
    cudaStream_t stream)
{
    dim3 block_dim(256);
    dim3 grid_dim; // Grid for activation backward and parts of rotational backward
    grid_dim.x = (N_out + block_dim.x - 1) / block_dim.x;
    grid_dim.y = batch_size;
    grid_dim.z = 1;

    // Call Activation Backward Kernel
    isokawaActivationBackwardKernel<<<grid_dim, block_dim, 0, stream>>>(
        d_grad_Y_ptr, d_S_internal_saved_ptr, d_grad_S_internal_ptr, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // Call Rotational Backward Kernel
    // The grid/block for rotational backward might need to be different for optimal
    // computation of dW and dX, especially if you parallelize differently.
    // For the current stub focusing on dTheta, this grid might be okay if N_out blocks are launched.
    dim3 grid_dim_rot_bw;
    grid_dim_rot_bw.x = N_out; // One block per output feature j for d_theta_q accumulation
    grid_dim_rot_bw.y = 1;     // Batch accumulation happens inside the kernel for d_theta_q
    grid_dim_rot_bw.z = 1;
    // A more general rotational backward might use a 2D or 3D grid based on N_out, M_in, batch_size.
    
    isokawaRotationalBackwardKernel<<<grid_dim_rot_bw, block_dim, 0, stream>>>(
        d_grad_S_internal_ptr, d_X_saved_ptr, d_W_saved_ptr,
        d_grad_X_ptr, d_grad_W_ptr, d_grad_theta_ptr,
        M_in, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());
}