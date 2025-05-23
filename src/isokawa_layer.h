// src/isokawa_layer.h
#pragma once
#include "quat_ops.h" // For Quaternion struct

// --- FORWARD PASS KERNELS ---

// Kernel for Isokawa rotational neuron computation (s_j part)
__global__ void isokawaRotationalForwardKernel(
    const Quaternion* __restrict__ W_q,           // Shape: (N_out, M_in, 4) or flattened
    const Quaternion* __restrict__ X_batch,     // Shape: (batch_size, M_in, 4)
    const Quaternion* __restrict__ theta_q,       // Shape: (N_out, 4)
    Quaternion* __restrict__ S_internal_batch,  // Output: (batch_size, N_out, 4)
    int M_in,
    int N_out,
    int batch_size);

// Kernel for Isokawa's activation function
__global__ void isokawaActivationKernel(
    const Quaternion* __restrict__ S_internal_batch, // Input: (batch_size, N_out, 4)
    Quaternion* __restrict__ Y_batch,                // Output: (batch_size, N_out, 4)
    int N_out,
    int batch_size);

// --- BACKWARD PASS KERNEL DECLARATIONS (STUBS for now, to be implemented) ---

// Kernel to compute dL/dS_internal from dL/dY
__global__ void isokawaActivationBackwardKernel(
    const Quaternion* __restrict__ grad_Y_batch,         // Gradient from next layer: (batch_size, N_out, 4)
    const Quaternion* __restrict__ S_internal_batch,     // Pre-activation values saved from forward: (batch_size, N_out, 4)
    Quaternion* __restrict__ grad_S_internal_batch,  // Output gradient: (batch_size, N_out, 4)
    int N_out,
    int batch_size);

// Kernel to compute dL/dX, dL/dW, dL/dTheta from dL/dS_internal
__global__ void isokawaRotationalBackwardKernel(
    const Quaternion* __restrict__ grad_S_internal_batch, // Gradient w.r.t. pre-threshold sum: (batch_size, N_out, 4)
    const Quaternion* __restrict__ X_batch,               // Inputs saved from forward: (batch_size, M_in, 4)
    const Quaternion* __restrict__ W_q,                   // Weights saved from forward: (N_out, M_in, 4)
    // Output gradients:
    Quaternion* __restrict__ grad_X_batch,            // Gradient w.r.t. inputs X: (batch_size, M_in, 4)
    Quaternion* __restrict__ grad_W_q,                // Gradient w.r.t. weights W: (N_out, M_in, 4)
    Quaternion* __restrict__ grad_theta_q,            // Gradient w.r.t. thresholds theta: (N_out, 4)
    int M_in,
    int N_out,
    int batch_size);