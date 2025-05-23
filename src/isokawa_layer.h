// src/isokawa_layer.h
#pragma once
#include "quat_ops.h" // For Quaternion struct
#include <torch/types.h> // For at::Tensor (forward declaration or include full if needed by host functions)

// --- FORWARD PASS KERNELS ---
// (Kernel declarations remain the same)
__global__ void isokawaRotationalForwardKernel(
    const Quaternion* __restrict__ W_q,
    const Quaternion* __restrict__ X_batch,
    const Quaternion* __restrict__ theta_q,
    Quaternion* __restrict__ S_internal_batch,
    int M_in,
    int N_out,
    int batch_size);

__global__ void isokawaActivationKernel(
    const Quaternion* __restrict__ S_internal_batch,
    Quaternion* __restrict__ Y_batch,
    int N_out,
    int batch_size);

// --- BACKWARD PASS KERNEL DECLARATIONS ---
// (Kernel declarations remain the same)
__global__ void isokawaActivationBackwardKernel(
    const Quaternion* __restrict__ grad_Y_batch,
    const Quaternion* __restrict__ S_internal_batch,
    Quaternion* __restrict__ grad_S_internal_batch,
    int N_out,
    int batch_size);

__global__ void isokawaRotationalBackwardKernel(
    const Quaternion* __restrict__ grad_S_internal_batch,
    const Quaternion* __restrict__ X_batch,
    const Quaternion* __restrict__ W_q,
    Quaternion* __restrict__ grad_X_batch,
    Quaternion* __restrict__ grad_W_q,
    Quaternion* __restrict__ grad_theta_q,
    int M_in,
    int N_out,
    int batch_size);

// --- C++ HOST FUNCTIONS TO LAUNCH KERNELS ---
// These will be defined in isokawa_layer.cu and called from bindings.cpp

void launch_isokawa_forward_kernels(
    const Quaternion* d_W_ptr,
    const Quaternion* d_X_ptr,
    const Quaternion* d_theta_ptr,
    Quaternion* d_S_internal_ptr,
    Quaternion* d_Y_ptr,
    int M_in,
    int N_out,
    int batch_size,
    cudaStream_t stream = 0); // Optional stream parameter

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
    cudaStream_t stream = 0); // Optional stream parameter