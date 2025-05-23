// src/bindings.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "quatnet_layer.h" // Your existing Parcollet-style QuaternionDenseLayer
#include "quat_ops.h"      // For Quaternion struct
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

// --- Global map to store layer instances ---
// This is one way to manage C++ object lifetime if PyTorch needs to refer to them.
// A better way for autograd is often to pass W and B tensors directly to stateless C++ functions.
// For now, we use this map to associate a Python layer instance with a C++ layer instance.
std::map<int64_t, std::shared_ptr<QuaternionDenseLayer>> layer_instances;
int64_t next_layer_id = 0;

int64_t create_quatnet_dense_layer_cpp(int input_dim_quats, int output_dim_quats, int activation_type_val) {
    ActivationType act_type = static_cast<ActivationType>(activation_type_val);
    std::shared_ptr<QuaternionDenseLayer> layer = std::make_shared<QuaternionDenseLayer>(input_dim_quats, output_dim_quats, act_type);
    int64_t current_id = next_layer_id++;
    layer_instances[current_id] = layer;
    return current_id;
}

void destroy_quatnet_dense_layer_cpp(int64_t layer_id) {
    layer_instances.erase(layer_id);
}

// Forward pass binding
torch::Tensor quatnet_dense_forward_cpp(
    int64_t layer_id,
    const torch::Tensor& x_batch_tensor) {

    auto it = layer_instances.find(layer_id);
    TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id for forward pass.");
    QuaternionDenseLayer* layer_ptr = it->second.get();

    int batch_size = x_batch_tensor.size(0);
    TORCH_CHECK(x_batch_tensor.size(1) == layer_ptr->get_input_dim(), "Input tensor M_in_q mismatch.");

    const Quaternion* d_X_ptr = get_const_q_tensor_ptr(x_batch_tensor);

    auto tensor_opts = torch::TensorOptions().device(x_batch_tensor.device()).dtype(torch::kFloat32);
    torch::Tensor y_out_tensor = torch::empty({batch_size, layer_ptr->get_output_dim(), 4}, tensor_opts);
    Quaternion* d_Y_ptr = get_q_tensor_ptr(y_out_tensor);
    
    // The C++ layer's forward method needs to be called. It might use its own internal stream.
    // PyTorch usually manages streams, so passing at::cuda::getCurrentCUDAStream() might be an option
    // if your layer's forward can take a stream. For now, assume default stream.
    layer_ptr->forward(d_X_ptr, d_Y_ptr, batch_size);
    // CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion before tensor is used by PyTorch.
                                        // PyTorch autograd usually handles this with its stream context.

    return y_out_tensor;
}

// Backward pass binding
// Returns grad_x_tensor. Gradients for W and B are accumulated in the C++ layer object.
torch::Tensor quatnet_dense_backward_cpp(
    int64_t layer_id,
    const torch::Tensor& grad_y_tensor,
    const torch::Tensor& x_saved_tensor // Input saved during forward pass
) {
    auto it = layer_instances.find(layer_id);
    TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id for backward pass.");
    QuaternionDenseLayer* layer_ptr = it->second.get();

    int batch_size = grad_y_tensor.size(0);
    TORCH_CHECK(grad_y_tensor.size(1) == layer_ptr->get_output_dim(), "grad_y_tensor N_out_q mismatch.");
    TORCH_CHECK(x_saved_tensor.size(1) == layer_ptr->get_input_dim(), "x_saved_tensor M_in_q mismatch.");
    TORCH_CHECK(x_saved_tensor.size(0) == batch_size, "Batch size mismatch for x_saved_tensor.");


    const Quaternion* d_grad_Y_ptr = get_const_q_tensor_ptr(grad_y_tensor);
    const Quaternion* d_X_saved_ptr = get_const_q_tensor_ptr(x_saved_tensor);

    auto tensor_opts = torch::TensorOptions().device(grad_y_tensor.device()).dtype(torch::kFloat32);
    torch::Tensor grad_x_tensor = torch::empty({batch_size, layer_ptr->get_input_dim(), 4}, tensor_opts);
    grad_x_tensor.zero_(); // Important if the kernel accumulates or doesn't write all parts
    Quaternion* d_grad_X_ptr = get_q_tensor_ptr(grad_x_tensor);

    layer_ptr->backward(d_grad_Y_ptr, d_X_saved_ptr, d_grad_X_ptr, batch_size);
    // CUDA_CHECK(cudaDeviceSynchronize());

    return grad_x_tensor;
}

// Functions to get/set weights and biases for synchronization with PyTorch nn.Parameter
torch::Tensor get_weights_cpp(int64_t layer_id) {
    auto it = layer_instances.find(layer_id);
    TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id for get_weights_cpp.");
    QuaternionDenseLayer* layer_ptr = it->second.get();
    
    int N_out = layer_ptr->get_output_dim();
    int M_in = layer_ptr->get_input_dim();
    auto tensor_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    torch::Tensor w_tensor = torch::empty({N_out, M_in, 4}, tensor_opts);
    CUDA_CHECK(cudaMemcpy(get_q_tensor_ptr(w_tensor), layer_ptr->get_weights_ptr(), 
                          N_out * M_in * sizeof(Quaternion), cudaMemcpyDeviceToDevice)); // Should be DeviceToHost then .to(device) or direct D2D if on same device
    return w_tensor; // This returns a copy. Needs careful handling for gradients.
}
// Similar functions for get_bias_cpp, set_weights_cpp, set_bias_cpp,
// zero_gradients_cpp(layer_id), update_weights_cpp(layer_id, lr) would be needed
// if parameters are managed this way. This gets complex with autograd.

// A simpler model for autograd: pass W and B tensors *to* stateless forward/backward C++ functions.
// The current C++ QuaternionDenseLayer is stateful. We will bind its methods.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Pybind11 bindings for QuatNet C++ QuaternionDenseLayer";

    py::enum_<ActivationType>(m, "ActivationType")
        .value("NONE", ActivationType::NONE)
        .value("SPLIT_TANH", ActivationType::SPLIT_TANH)
        .value("SPLIT_SIGMOID", ActivationType::SPLIT_SIGMOID)
        .value("PURE_IMAG_SIGMOID", ActivationType::PURE_IMAG_SIGMOID)
        .export_values();
    
    // These functions interact with the layer_instances map
    m.def("create_dense_layer", &create_quatnet_dense_layer_cpp, "Create QuaternionDenseLayer C++ object");
    m.def("destroy_dense_layer", &destroy_quatnet_dense_layer_cpp, "Destroy QuaternionDenseLayer C++ object");
    m.def("dense_forward", &quatnet_dense_forward_cpp, "QuaternionDenseLayer Forward for PyTorch");
    m.def("dense_backward", &quatnet_dense_backward_cpp, "QuaternionDenseLayer Backward for PyTorch");
    
    // Need functions to access gradients stored in C++ layer and to update weights
    m.def("get_weight_gradient_cpp", [](int64_t layer_id) {
        auto it = layer_instances.find(layer_id);
        TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id.");
        QuaternionDenseLayer* layer = it->second.get();
        // Assuming d_grad_W is accessible or you add a getter
        // This is simplified; d_grad_W needs to be exposed carefully.
        // For now, this is a placeholder for how PyTorch would get grads.
        auto tensor_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        torch::Tensor grad_w_tensor = torch::empty({layer->get_output_dim(), layer->get_input_dim(), 4}, tensor_opts);
        // CUDA_CHECK(cudaMemcpy(get_q_tensor_ptr(grad_w_tensor), layer->get_grad_weights_ptr(), ...)); // Requires get_grad_weights_ptr()
        return grad_w_tensor; // Placeholder
    });
     m.def("get_bias_gradient_cpp", [](int64_t layer_id) {
        auto it = layer_instances.find(layer_id);
        TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id.");
        QuaternionDenseLayer* layer = it->second.get();
        auto tensor_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        torch::Tensor grad_b_tensor = torch::empty({layer->get_output_dim(), 4}, tensor_opts);
        // CUDA_CHECK(cudaMemcpy(get_q_tensor_ptr(grad_b_tensor), layer->get_grad_bias_ptr(), ...)); // Requires get_grad_bias_ptr()
        return grad_b_tensor; // Placeholder
    });
    m.def("update_weights_cpp", [](int64_t layer_id, float lr){
        auto it = layer_instances.find(layer_id);
        TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id.");
        it->second->update_weights(lr);
    });
    m.def("zero_gradients_cpp", [](int64_t layer_id){
        auto it = layer_instances.find(layer_id);
        TORCH_CHECK(it != layer_instances.end(), "Invalid layer_id.");
        it->second->zero_gradients();
    });
}