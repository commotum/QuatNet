// src/bindings.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "quatnet_layer.h" // Your existing Parcollet-style layer
#include "quat_ops.h"      // For Quaternion struct definition
#include "utils.h"         // For CUDA_CHECK

// Helper to get const Quaternion data pointer
const Quaternion* get_const_q_tensor_ptr(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Input tensor must be a CUDA tensor. Got: ", tensor.toString());
    TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous. Got: ", tensor.toString());
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Input tensor must be Float32. Got: ", tensor.toString());
    TORCH_CHECK(tensor.size(-1) == 4, "Input tensor's last dimension must be 4. Got: ", tensor.toString());
    return reinterpret_cast<const Quaternion*>(tensor.data_ptr<float>());
}

// Helper to get mutable Quaternion data pointer
Quaternion* get_q_tensor_ptr(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Output tensor must be a CUDA tensor. Got: ", tensor.toString());
    TORCH_CHECK(tensor.is_contiguous(), "Output tensor must be contiguous. Got: ", tensor.toString());
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Output tensor must be Float32. Got: ", tensor.toString());
    TORCH_CHECK(tensor.size(-1) == 4, "Output tensor's last dimension must be 4. Got: ", tensor.toString());
    return reinterpret_cast<Quaternion*>(tensor.data_ptr<float>());
}


// Wrapper for QuaternionDenseLayer forward
// Instead of returning S_internal, we'll rely on the C++ layer to store it if needed for its own backward.
torch::Tensor quatnet_dense_forward_cpp(
    QuaternionDenseLayer* layer_ptr, // Pass pointer to the C++ layer object
    const torch::Tensor& x_batch_tensor) {

    TORCH_CHECK(layer_ptr != nullptr, "QuaternionDenseLayer pointer is null.");
    int batch_size = x_batch_tensor.size(0);
    int M_in_q = x_batch_tensor.size(1);
    TORCH_CHECK(M_in_q == layer_ptr->get_input_dim(), "Input tensor M_in_q mismatch with layer input_dim.");

    const Quaternion* d_X_ptr = get_const_q_tensor_ptr(x_batch_tensor);

    auto tensor_opts = torch::TensorOptions().device(x_batch_tensor.device()).dtype(torch::kFloat32);
    torch::Tensor y_out_tensor = torch::empty({batch_size, layer_ptr->get_output_dim(), 4}, tensor_opts);
    Quaternion* d_Y_ptr = get_q_tensor_ptr(y_out_tensor);

    // The C++ layer's forward method should handle streams internally if necessary or use default.
    layer_ptr->forward(d_X_ptr, d_Y_ptr, batch_size);
    // No explicit stream sync here, assume PyTorch handles it or layer does.

    return y_out_tensor;
}

// Wrapper for QuaternionDenseLayer backward
torch::Tensor quatnet_dense_backward_cpp(
    QuaternionDenseLayer* layer_ptr, // Pointer to the C++ layer object
    const torch::Tensor& grad_y_tensor,
    const torch::Tensor& x_saved_tensor // Input saved during forward pass
) {
    TORCH_CHECK(layer_ptr != nullptr, "QuaternionDenseLayer pointer is null.");
    int batch_size = grad_y_tensor.size(0);
    int N_out_q = grad_y_tensor.size(1);
    int M_in_q = x_saved_tensor.size(1);

    TORCH_CHECK(N_out_q == layer_ptr->get_output_dim(), "grad_y_tensor N_out_q mismatch.");
    TORCH_CHECK(M_in_q == layer_ptr->get_input_dim(), "x_saved_tensor M_in_q mismatch.");


    const Quaternion* d_grad_Y_ptr = get_const_q_tensor_ptr(grad_y_tensor);
    const Quaternion* d_X_saved_ptr = get_const_q_tensor_ptr(x_saved_tensor);

    auto tensor_opts = torch::TensorOptions().device(grad_y_tensor.device()).dtype(torch::kFloat32);
    torch::Tensor grad_x_tensor = torch::empty({batch_size, M_in_q, 4}, tensor_opts);
    Quaternion* d_grad_X_ptr = get_q_tensor_ptr(grad_x_tensor);

    // The C++ layer's backward method computes dL/dW, dL/dB (internally) and dL/dX (output here)
    layer_ptr->backward(d_grad_Y_ptr, d_X_saved_ptr, d_grad_X_ptr, batch_size);
    
    // Gradients for W and B are accumulated inside the layer_ptr object.
    // We only need to return grad_x_tensor for the previous layer.
    return grad_x_tensor;
}

// Helper to get weight and bias tensors from the C++ layer (for PyTorch parameters)
// This approach assumes parameters are primarily managed by the C++ object.
// For nn.Parameters in PyTorch, you'd copy from PyTorch params to C++ layer or vice-versa.
// Let's assume for now we want to expose the C++ layer's weights as non-trainable Tensors
// OR, if they are to be nn.Parameters, the PyTorch layer needs to handle syncing them.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Pybind11 bindings for QuatNet C++ Layers";

    py::class_<QuaternionDenseLayer>(m, "QuaternionDenseLayerCpp")
        .def(py::init<int, int, ActivationType>(), 
             py::arg("input_dim_quats"), py::arg("output_dim_quats"), 
             py::arg("act_type") = ActivationType::NONE)
        // Expose methods needed by autograd.Function or for direct use
        // The forward/backward here are the C++ class methods, not directly for autograd.
        // .def("forward", &QuaternionDenseLayer::forward) // Problematic with raw pointers
        // .def("backward", &QuaternionDenseLayer::backward) // Problematic
        .def("update_weights", &QuaternionDenseLayer::update_weights, py::arg("learning_rate"))
        .def("zero_gradients", &QuaternionDenseLayer::zero_gradients)
        .def("get_input_dim", &QuaternionDenseLayer::get_input_dim)
        .def("get_output_dim", &QuaternionDenseLayer::get_output_dim)
        // Methods to get pointers to weights/biases for autograd to wrap them
        // This is tricky because nn.Parameter needs to *own* the data.
        // A better way for autograd: pass W and B tensors *into* C++ functions.
        // The current C++ QuaternionDenseLayer manages its own d_W and d_b.
        // For autograd.Function, we'll need dedicated C++ functions that take W and B as tensors.
        ;

    py::enum_<ActivationType>(m, "ActivationType")
        .value("NONE", ActivationType::NONE)
        .value("SPLIT_TANH", ActivationType::SPLIT_TANH)
        .value("SPLIT_SIGMOID", ActivationType::SPLIT_SIGMOID)
        .value("PURE_IMAG_SIGMOID", ActivationType::PURE_IMAG_SIGMOID)
        .export_values();
    
    // These are the functions that the autograd.Function will call
    m.def("quatnet_dense_forward_wrapper", &quatnet_dense_forward_cpp, "QuaternionDenseLayer Forward for PyTorch");
    m.def("quatnet_dense_backward_wrapper", &quatnet_dense_backward_cpp, "QuaternionDenseLayer Backward for PyTorch");
}