// src/quatnet_layer.h (Modify or ensure it has these features)
#pragma once
#include "quat_ops.h" // Use your updated quat_ops.h
#include "utils.h"    // For CUDA_CHECK
#include <curand.h>   // If using cuRAND for initialization

// Enum to specify activation type
enum class ActivationType {
    NONE,        // Outputs pre-activations
    SPLIT_TANH,  // Component-wise tanh on all 4 parts
    SPLIT_SIGMOID, // Component-wise sigmoid on all 4 parts
    PURE_IMAG_SIGMOID // Sigmoid on i,j,k parts; real part is 0
};

class QuaternionDenseLayer {
private:
    Quaternion* d_W;     // Weights on device, size N_out * M_in
    Quaternion* d_b;     // Biases on device, size N_out
    int M_in_q, N_out_q; // Input and Output quaternion dimensions
    ActivationType activation_type;

    // For storing pre-activations if needed for backward pass with certain activation handling
    Quaternion* d_pre_activations; // Buffer of size batch_size * N_out_q
    // For storing inputs if needed for backward pass
    const Quaternion* d_last_input_batch; // Pointer, not owned, size batch_size * M_in_q
    int last_batch_size;

    // Gradients (accumulated here)
    Quaternion* d_grad_W;
    Quaternion* d_grad_b;

    curandGenerator_t curand_gen; // If using cuRAND for initialization

    void initializeParameters();
    void apply_activation_forward(Quaternion* pre_act, Quaternion* post_act, int count);
    void apply_activation_backward(Quaternion* grad_post_act, const Quaternion* pre_act_saved, 
                                   Quaternion* grad_pre_act, int count);


public:
    QuaternionDenseLayer(int input_dim_quats, int output_dim_quats, ActivationType act_type = ActivationType::NONE);
    ~QuaternionDenseLayer();

    // Forward pass: output = Activation( (W @ input) + B )
    // If act_type is NONE, output contains pre-activations.
    // `input_batch` shape: (batch_size, M_in_q, 4)
    // `output_batch` shape: (batch_size, N_out_q, 4)
    void forward(const Quaternion* input_batch, Quaternion* output_batch, int batch_size);

    // Backward pass:
    // Computes dL/dW, dL/dB (accumulated internally) and dL/dX (returned in grad_input_batch)
    // `grad_output_batch` is dL/d(output_of_this_layer), shape (batch_size, N_out_q, 4)
    // `grad_input_batch` is dL/d(input_to_this_layer), shape (batch_size, M_in_q, 4)
    void backward(const Quaternion* grad_output_batch, // Grad w.r.t. layer's output (post-activation)
                  const Quaternion* input_batch_saved, // Input to this layer during forward
                  Quaternion* grad_input_batch,      // To store dL/dX for previous layer
                  int batch_size);

    void update_weights(float learning_rate);
    void zero_gradients(); // Helper to zero out d_grad_W and d_grad_b

    // To get parameters for PyTorch wrapper
    Quaternion* get_weights_ptr() { return d_W; }
    Quaternion* get_bias_ptr() { return d_b; }
    int get_input_dim() const { return M_in_q; }
    int get_output_dim() const { return N_out_q; }
    int get_num_weight_params() const { return M_in_q * N_out_q * 4; }
    int get_num_bias_params() const { return N_out_q * 4; }
};