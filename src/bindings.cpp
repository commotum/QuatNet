// src/bindings.cpp
#include <torch/extension.h>
#include "isokawa_layer.h" // For kernel declarations
#include "quat_ops.h"      // For Quaternion struct definition
#include "utils.h"         // For CUDA_CHECK

// Helper to get const Quaternion data pointer and check tensor properties
const Quaternion* get_const_q_tensor_ptr(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Input tensor must be a CUDA tensor. Got: ", tensor.toString());
    TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous. Got: ", tensor.toString());
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Input tensor must be Float32. Got: ", tensor.toString());
    TORCH_CHECK(tensor.size(-1) == 4, "Input tensor's last dimension must be 4 (quaternion components). Got: ", tensor.toString());
    return reinterpret_cast<const Quaternion*>(tensor.data_ptr<float>());
}

// Helper to get mutable Quaternion data pointer
Quaternion* get_q_tensor_ptr(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Output tensor must be a CUDA tensor. Got: ", tensor.toString());
    TORCH_CHECK(tensor.is_contiguous(), "Output tensor must be contiguous. Got: ", tensor.toString());
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Output tensor must be Float32. Got: ", tensor.toString());
    TORCH_CHECK(tensor.size(-1) == 4, "Output tensor's last dimension must be 4 (quaternion components). Got: ", tensor.toString());
    return reinterpret_cast<Quaternion*>(tensor.data_ptr<float>());
}

// Forward pass binding for autograd.Function
// Returns a pair: {output_Y_tensor, S_internal_tensor}
std::pair<torch::Tensor, torch::Tensor> isokawa_forward_cpp(
        const torch::Tensor& x_batch_tensor, // (batch, M_in, 4)
        const torch::Tensor& w_q_tensor,     // (N_out, M_in, 4)
        const torch::Tensor& theta_q_tensor, // (N_out, 4)
        int M_in, int N_out) {

    int batch_size = x_batch_tensor.size(0);
    TORCH_CHECK(M_in == x_batch_tensor.size(1), "x_batch_tensor M_in dimension mismatch.");
    TORCH_CHECK(N_out == w_q_tensor.size(0), "w_q_tensor N_out dimension mismatch.");
    TORCH_CHECK(M_in == w_q_tensor.size(1), "w_q_tensor M_in dimension mismatch.");
    TORCH_CHECK(N_out == theta_q_tensor.size(0), "theta_q_tensor N_out dimension mismatch.");


    const Quaternion* d_X_ptr = get_const_q_tensor_ptr(x_batch_tensor);
    const Quaternion* d_W_ptr = get_const_q_tensor_ptr(w_q_tensor);
    const Quaternion* d_theta_ptr = get_const_q_tensor_ptr(theta_q_tensor);

    auto tensor_opts = torch::TensorOptions().device(x_batch_tensor.device()).dtype(torch::kFloat32);
    torch::Tensor s_internal_tensor = torch::empty({batch_size, N_out, 4}, tensor_opts);
    torch::Tensor y_out_tensor = torch::empty({batch_size, N_out, 4}, tensor_opts);

    Quaternion* d_S_internal_ptr = get_q_tensor_ptr(s_internal_tensor);
    Quaternion* d_Y_ptr = get_q_tensor_ptr(y_out_tensor);

    dim3 block_dim(256); // Example block size, can be tuned
    dim3 grid_dim;
    grid_dim.x = (N_out + block_dim.x - 1) / block_dim.x;
    grid_dim.y = batch_size;
    grid_dim.z = 1;

    isokawaRotationalForwardKernel<<<grid_dim, block_dim>>>(
        d_W_ptr, d_X_ptr, d_theta_ptr, d_S_internal_ptr, M_in, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch

    isokawaActivationKernel<<<grid_dim, block_dim>>>(
        d_S_internal_ptr, d_Y_ptr, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());

    return {y_out_tensor, s_internal_tensor};
}

// Backward pass binding for autograd.Function (STUB - IMPLEMENT THIS LATER)
// Returns {grad_X, grad_W, grad_theta}
std::vector<torch::Tensor> isokawa_backward_cpp(
    const torch::Tensor& grad_y_tensor,           // (batch, N_out, 4)
    const torch::Tensor& x_saved_tensor,          // (batch, M_in, 4)
    const torch::Tensor& w_saved_tensor,          // (N_out, M_in, 4)
    const torch::Tensor& theta_saved_tensor,      // (N_out, 4)
    const torch::Tensor& s_internal_saved_tensor, // (batch, N_out, 4)
    int M_in, int N_out
) {
    int batch_size = grad_y_tensor.size(0);

    const Quaternion* d_grad_Y_ptr = get_const_q_tensor_ptr(grad_y_tensor);
    const Quaternion* d_X_saved_ptr = get_const_q_tensor_ptr(x_saved_tensor);
    const Quaternion* d_W_saved_ptr = get_const_q_tensor_ptr(w_saved_tensor);
    // theta_saved_tensor is not directly used in backward for S_internal, but dL/dTheta is an output
    const Quaternion* d_S_internal_saved_ptr = get_const_q_tensor_ptr(s_internal_saved_tensor);


    auto tensor_opts = torch::TensorOptions().device(grad_y_tensor.device()).dtype(torch::kFloat32);
    torch::Tensor grad_s_internal_tensor = torch::empty({batch_size, N_out, 4}, tensor_opts);
    torch::Tensor grad_x_tensor = torch::empty_like(x_saved_tensor); // Or zeros_like if kernel doesn't write all
    torch::Tensor grad_w_tensor = torch::empty_like(w_saved_tensor); // Or zeros_like and use atomicAdd in kernel
    torch::Tensor grad_theta_tensor = torch::empty_like(theta_saved_tensor); // Or zeros_like

    Quaternion* d_grad_S_internal_ptr = get_q_tensor_ptr(grad_s_internal_tensor);
    Quaternion* d_grad_X_ptr = get_q_tensor_ptr(grad_x_tensor);
    Quaternion* d_grad_W_ptr = get_q_tensor_ptr(grad_w_tensor);
    Quaternion* d_grad_theta_ptr = get_q_tensor_ptr(grad_theta_tensor);

    // Initialize output gradient tensors to zero if kernels accumulate
    grad_x_tensor.zero_();
    grad_w_tensor.zero_();
    grad_theta_tensor.zero_();


    dim3 block_dim(256);
    dim3 grid_dim;
    grid_dim.x = (N_out + block_dim.x - 1) / block_dim.x;
    grid_dim.y = batch_size; // Or adjust if kernels parallelize over batch differently
    grid_dim.z = 1;

    // Call Activation Backward Kernel
    isokawaActivationBackwardKernel<<<grid_dim, block_dim>>>(
        d_grad_Y_ptr, d_S_internal_saved_ptr, d_grad_S_internal_ptr, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // Call Rotational Backward Kernel
    // Grid/block might need adjustment for grad_W, grad_X computations
    // This is a simplified launch assuming grad_theta_q is handled per N_out and accumulates batch.
    // For grad_W and grad_X, more complex parallelization is needed.
    // The stub kernel for rotational backward needs significant work.
    isokawaRotationalBackwardKernel<<<grid_dim, block_dim>>>( // May need different grid/block for dW, dX
        d_grad_S_internal_ptr, d_X_saved_ptr, d_W_saved_ptr,
        d_grad_X_ptr, d_grad_W_ptr, d_grad_theta_ptr,
        M_in, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());

    return {grad_x_tensor, grad_w_tensor, grad_theta_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Pybind11 bindings for Isokawa Quaternion Layer CUDA Kernels";
    m.def("isokawa_forward_cpp", &isokawa_forward_cpp, "Isokawa Layer Forward (C++ backend)",
          py::arg("x_batch_tensor"), py::arg("w_q_tensor"), py::arg("theta_q_tensor"),
          py::arg("M_in"), py::arg("N_out"));
    m.def("isokawa_backward_cpp", &isokawa_backward_cpp, "Isokawa Layer Backward (C++ backend - STUB)",
          py::arg("grad_y_tensor"), py::arg("x_saved_tensor"), py::arg("w_saved_tensor"),
          py::arg("theta_saved_tensor"), py::arg("s_internal_saved_tensor"),
          py::arg("M_in"), py::arg("N_out"));
}