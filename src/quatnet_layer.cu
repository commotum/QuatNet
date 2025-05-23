// src/quatnet_layer.cu
#include "quatnet_layer.h"
#include "hamprod_kernel.h" // For launchQuatMatVecMul, launchAddBias
#include <cmath>           // For tanhf, expf
#include <stdexcept>
#include <vector>
#include <cstdlib>        // For rand, srand
#include <ctime>          // For time(0)

// --- Helper Kernels for Activations ---
__global__ void apply_activation_forward_kernel(
    const Quaternion* pre_act, Quaternion* post_act, int count, ActivationType act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        Quaternion p = pre_act[idx];
        Quaternion result;
        switch (act_type) {
            case ActivationType::SPLIT_TANH:
                result = {device_component_tanh(p.w), device_component_tanh(p.x), device_component_tanh(p.y), device_component_tanh(p.z)};
                break;
            case ActivationType::SPLIT_SIGMOID:
                result = {device_component_sigmoid(p.w), device_component_sigmoid(p.x), device_component_sigmoid(p.y), device_component_sigmoid(p.z)};
                break;
            case ActivationType::PURE_IMAG_SIGMOID:
                result = {0.0f, device_component_sigmoid(p.x), device_component_sigmoid(p.y), device_component_sigmoid(p.z)};
                break;
            case ActivationType::NONE:
            default:
                result = p;
                break;
        }
        post_act[idx] = result;
    }
}

__global__ void apply_activation_backward_kernel(
    const Quaternion* grad_post_act, const Quaternion* pre_act_saved,
    Quaternion* grad_pre_act, int count, ActivationType act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        Quaternion g_post = grad_post_act[idx];
        Quaternion p_saved = pre_act_saved[idx]; // Pre-activation values
        Quaternion result_grad_pre;

        switch (act_type) {
            case ActivationType::SPLIT_TANH: {
                float tanh_pw = device_component_tanh(p_saved.w); float dtanh_pw = 1.0f - tanh_pw * tanh_pw;
                float tanh_px = device_component_tanh(p_saved.x); float dtanh_px = 1.0f - tanh_px * tanh_px;
                float tanh_py = device_component_tanh(p_saved.y); float dtanh_py = 1.0f - tanh_py * tanh_py;
                float tanh_pz = device_component_tanh(p_saved.z); float dtanh_pz = 1.0f - tanh_pz * tanh_pz;
                result_grad_pre = {g_post.w * dtanh_pw, g_post.x * dtanh_px, g_post.y * dtanh_py, g_post.z * dtanh_pz};
                break;
            }
            case ActivationType::SPLIT_SIGMOID: {
                float sig_pw = device_component_sigmoid(p_saved.w); float dsig_pw = sig_pw * (1.0f - sig_pw);
                float sig_px = device_component_sigmoid(p_saved.x); float dsig_px = sig_px * (1.0f - sig_px);
                float sig_py = device_component_sigmoid(p_saved.y); float dsig_py = sig_py * (1.0f - sig_py);
                float sig_pz = device_component_sigmoid(p_saved.z); float dsig_pz = sig_pz * (1.0f - sig_pz);
                result_grad_pre = {g_post.w * dsig_pw, g_post.x * dsig_px, g_post.y * dsig_py, g_post.z * dsig_pz};
                break;
            }
            case ActivationType::PURE_IMAG_SIGMOID: {
                float sig_px = device_component_sigmoid(p_saved.x); float dsig_px = sig_px * (1.0f - sig_px);
                float sig_py = device_component_sigmoid(p_saved.y); float dsig_py = sig_py * (1.0f - sig_py);
                float sig_pz = device_component_sigmoid(p_saved.z); float dsig_pz = sig_pz * (1.0f - sig_pz);
                // Gradient for real part is 0 as activation output real part is 0 and not dependent on pre_act.w
                result_grad_pre = {0.0f, g_post.x * dsig_px, g_post.y * dsig_py, g_post.z * dsig_pz};
                break;
            }
            case ActivationType::NONE:
            default:
                result_grad_pre = g_post; // Pass through gradient if no activation
                break;
        }
        grad_pre_act[idx] = result_grad_pre;
    }
}

// --- QuaternionDenseLayer Methods ---
void QuaternionDenseLayer::initializeParameters() {
    float fan_in = static_cast<float>(M_in_q);
    float fan_out = static_cast<float>(N_out_q);
    float sigma_sq = 2.0f / (fan_in + fan_out); // Xavier/Glorot variance
    float sigma = std::sqrt(sigma_sq);

    std::vector<float> h_W_floats(N_out_q * M_in_q * 4);
    std::vector<float> h_b_floats(N_out_q * 4);
    
    srand(static_cast<unsigned int>(time(nullptr)));
    for(size_t i = 0; i < h_W_floats.size(); ++i) {
        // Using Box-Muller for a slightly better normal distribution
        float u1 = static_cast<float>(rand() + 1) / (RAND_MAX + 1.0f); // Avoid log(0)
        float u2 = static_cast<float>(rand() + 1) / (RAND_MAX + 1.0f);
        h_W_floats[i] = sigma * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    }
    for(size_t i = 0; i < h_b_floats.size(); ++i) {
        // Biases often initialized to zero or small constant/random value
        h_b_floats[i] = 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }

    CUDA_CHECK(cudaMemcpy(d_W, h_W_floats.data(), h_W_floats.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_floats.data(), h_b_floats.size() * sizeof(float), cudaMemcpyHostToDevice));
}

QuaternionDenseLayer::QuaternionDenseLayer(int input_dim_quats, int output_dim_quats, ActivationType act_type)
    : M_in_q(input_dim_quats), N_out_q(output_dim_quats), activation_type(act_type),
      d_pre_activations(nullptr), d_last_input_batch(nullptr), last_batch_size(0),
      d_grad_W(nullptr), d_grad_b(nullptr) {
    
    CUDA_CHECK(cudaMalloc(&d_W, N_out_q * M_in_q * sizeof(Quaternion)));
    CUDA_CHECK(cudaMalloc(&d_b, N_out_q * sizeof(Quaternion)));
    
    initializeParameters(); // Using host-side RNG and copy

    // Allocate gradient buffers
    CUDA_CHECK(cudaMalloc(&d_grad_W, N_out_q * M_in_q * sizeof(Quaternion)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, N_out_q * sizeof(Quaternion)));
    zero_gradients(); // Initialize gradients to zero
}

QuaternionDenseLayer::~QuaternionDenseLayer() {
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_grad_W);
    cudaFree(d_grad_b);
    if (d_pre_activations) cudaFree(d_pre_activations); // Free if allocated
}

void QuaternionDenseLayer::apply_activation_forward(Quaternion* pre_act, Quaternion* post_act, int count) {
    if (activation_type == ActivationType::NONE) {
        if (pre_act != post_act) { // Only copy if different buffers
            CUDA_CHECK(cudaMemcpy(post_act, pre_act, count * sizeof(Quaternion), cudaMemcpyDeviceToDevice));
        }
        return;
    }
    dim3 blockDim(256);
    dim3 gridDim((count + blockDim.x - 1) / blockDim.x);
    apply_activation_forward_kernel<<<gridDim, blockDim>>>(pre_act, post_act, count, activation_type);
    CUDA_CHECK(cudaGetLastError());
}

void QuaternionDenseLayer::apply_activation_backward(
    Quaternion* grad_post_act, const Quaternion* pre_act_saved, 
    Quaternion* grad_pre_act, int count) {
    if (activation_type == ActivationType::NONE) {
         if (grad_post_act != grad_pre_act) { // Only copy if different buffers
            CUDA_CHECK(cudaMemcpy(grad_pre_act, grad_post_act, count * sizeof(Quaternion), cudaMemcpyDeviceToDevice));
        }
        return;
    }
    dim3 blockDim(256);
    dim3 gridDim((count + blockDim.x - 1) / blockDim.x);
    apply_activation_backward_kernel<<<gridDim, blockDim>>>(grad_post_act, pre_act_saved, grad_pre_act, count, activation_type);
    CUDA_CHECK(cudaGetLastError());
}


void QuaternionDenseLayer::forward(const Quaternion* input_batch, Quaternion* output_batch, int batch_size) {
    // Allocate/Reallocate pre_activations buffer if batch_size changed or not allocated
    if (last_batch_size != batch_size || !d_pre_activations) {
        if (d_pre_activations) cudaFree(d_pre_activations);
        CUDA_CHECK(cudaMalloc(&d_pre_activations, batch_size * N_out_q * sizeof(Quaternion)));
        last_batch_size = batch_size;
    }
    d_last_input_batch = input_batch; // Save for backward pass (pointer only)

    // Each output sample in the batch is W @ input_sample + B
    for (int b = 0; b < batch_size; ++b) {
        const Quaternion* current_input_sample = input_batch + b * M_in_q;
        Quaternion* current_pre_activation_sample = d_pre_activations + b * N_out_q;
        
        // 1. W @ X (Matrix-vector multiply using Hamilton products for each sample)
        // launchQuatMatVecMul is from your hamprod_kernel.h, expecting (W, X_vec, OUT_vec, N_rows_W, N_cols_W)
        launchQuatMatVecMul(d_W, current_input_sample, current_pre_activation_sample, N_out_q, M_in_q);
        
        // 2. Add Bias B
        launchAddBias(current_pre_activation_sample, d_b, N_out_q);
    }
    // Wait for all WX+B to complete before activation
    CUDA_CHECK(cudaDeviceSynchronize());


    // 3. Apply Activation
    apply_activation_forward(d_pre_activations, output_batch, batch_size * N_out_q);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel for dL/dX = (dL/d(WX+B)) @ W^T (conceptually)
// For q_out = sum(w_k * q_in_k), dL/dq_in_k = sum_j dL/dq_out_j * w_jk^* (conjugate for transpose like effect)
// This needs to be implemented based on Parcollet Appendix 6.3 for input gradients.
__global__ void calculate_grad_input_kernel(
    const Quaternion* grad_pre_act_batch, // dL/d(WX+B), shape (batch_size, N_out_q)
    const Quaternion* W,                  // Weights (N_out_q, M_in_q)
    Quaternion* grad_input_batch,         // Output dL/dX, shape (batch_size, M_in_q) (zero-initialized)
    int M_in_q, int N_out_q, int batch_size) {

    int b = blockIdx.y; // Batch index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Input feature index (0 to M_in_q - 1)

    if (b >= batch_size || i >= M_in_q) return;

    Quaternion grad_x_i = {0,0,0,0};
    for (int j = 0; j < N_out_q; ++j) { // Sum over output features j
        Quaternion w_ji = W[j * M_in_q + i]; // W_ji is weight from input i to output j
        Quaternion grad_s_j = grad_pre_act_batch[b * N_out_q + j]; // dL/dS_j for current batch sample
        // dL/dX_i += dL/dS_j * W_ji^T  => In quaternion: conj(W_ji) * dL/dS_j
        // Or based on Parcollet: dL/dX_i += W_ji^* * dL/dS_j (careful with order)
        // Parcollet Eq for dE/dh_{t-1}: W_hh^* @ delta_n. This implies W_ji^* @ grad_S_j
        grad_x_i = grad_x_i + (w_ji.conjugate() * grad_s_j);
    }
    // grad_input_batch is zero-initialized, so direct write is fine if one thread per (b,i)
    // If accumulating via atomicAdd:
    atomicAddQuaternion(&grad_input_batch[b * M_in_q + i], grad_x_i);
}

// Kernel for dL/dW_ji = (dL/d(WX+B)_j) @ X_i^T (conceptually)
// dL/dW_ji = grad_S_j * conj(X_i)
__global__ void accumulate_grad_weights_kernel(
    const Quaternion* grad_pre_act_batch, // dL/d(WX+B), shape (batch_size, N_out_q)
    const Quaternion* input_batch_saved,  // X, shape (batch_size, M_in_q)
    Quaternion* grad_W,                   // Output dL/dW (N_out_q, M_in_q) (zero-initialized for accumulation)
    int M_in_q, int N_out_q, int batch_size) {

    // Each thread updates one W_ji, accumulating over the batch
    int ji_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = ji_idx / M_in_q; // Output neuron index
    int i = ji_idx % M_in_q; // Input neuron index

    if (j >= N_out_q || i >= M_in_q) return;

    Quaternion grad_w_ji_sum = {0,0,0,0};
    for (int b = 0; b < batch_size; ++b) {
        Quaternion grad_s_j = grad_pre_act_batch[b * N_out_q + j]; // dL/dS_j
        Quaternion x_i = input_batch_saved[b * M_in_q + i];    // X_i
        // dL/dW_ji += dL/dS_j * conj(X_i) (Parcollet Eq for W_hx: delta_n @ x_m^*)
        grad_w_ji_sum = grad_w_ji_sum + (grad_s_j * x_i.conjugate());
    }
    atomicAddQuaternion(&grad_W[j * M_in_q + i], grad_w_ji_sum);
}

// Kernel for dL/dB_j = dL/d(WX+B)_j
__global__ void accumulate_grad_bias_kernel(
    const Quaternion* grad_pre_act_batch, // dL/d(WX+B), shape (batch_size, N_out_q)
    Quaternion* grad_b,                   // Output dL/dB (N_out_q) (zero-initialized for accumulation)
    int N_out_q, int batch_size) {

    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output neuron index (bias index)
    if (j >= N_out_q) return;

    Quaternion grad_b_j_sum = {0,0,0,0};
    for (int b = 0; b < batch_size; ++b) {
        grad_b_j_sum = grad_b_j_sum + grad_pre_act_batch[b * N_out_q + j];
    }
    atomicAddQuaternion(&grad_b[j], grad_b_j_sum);
}


void QuaternionDenseLayer::backward(
    const Quaternion* grad_output_batch,      // Grad w.r.t. layer's output Y (post-activation)
    const Quaternion* input_batch_saved,      // Input X to this layer during forward (d_last_input_batch)
    Quaternion* grad_input_batch,           // To store dL/dX for previous layer (zero-initialized by caller)
    int batch_size) {

    if (!input_batch_saved) { // Should have been saved by forward pass
        throw std::runtime_error("Backward called without saving input_batch from forward pass or batch size mismatch.");
    }
    if (batch_size != last_batch_size) {
         throw std::runtime_error("Batch size in backward does not match forward pass batch size.");
    }


    // Buffer for dL/d(PreActivation) = dL/dS
    Quaternion* d_grad_pre_activations;
    CUDA_CHECK(cudaMalloc(&d_grad_pre_activations, batch_size * N_out_q * sizeof(Quaternion)));

    // 1. Backpropagate through activation function: get dL/dS from dL/dY
    //    dL/dS = (dL/dY) * f'(S)
    apply_activation_backward(const_cast<Quaternion*>(grad_output_batch), d_pre_activations, 
                              d_grad_pre_activations, batch_size * N_out_q);
    CUDA_CHECK(cudaDeviceSynchronize());

    // d_grad_pre_activations now holds dL/d(WX+B)

    // 2. Calculate gradient w.r.t. input X (dL/dX) for the previous layer
    //    dL/dX_i = sum_j (dL/dS_j * W_ji^T) -> sum_j (conj(W_ji) * dL/dS_j) or (W_ji_conj @ dL/dS_j)
    //    grad_input_batch should be zero-initialized by the caller if it's not null
    if (grad_input_batch) {
        dim3 block_dim_grad_x(256);
        dim3 grid_dim_grad_x((M_in_q + block_dim_grad_x.x -1)/block_dim_grad_x.x, batch_size, 1);
        calculate_grad_input_kernel<<<grid_dim_grad_x, block_dim_grad_x>>>(
            d_grad_pre_activations, d_W, grad_input_batch, M_in_q, N_out_q, batch_size
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. Accumulate gradients for weights W (dL/dW)
    //    dL/dW_ji = sum_batch (dL/dS_j * X_i^T) -> sum_batch (dL/dS_j * conj(X_i))
    int num_weights = N_out_q * M_in_q;
    dim3 block_dim_grad_w(256);
    dim3 grid_dim_grad_w((num_weights + block_dim_grad_w.x -1)/block_dim_grad_w.x, 1, 1);
    accumulate_grad_weights_kernel<<<grid_dim_grad_w, block_dim_grad_w>>>(
        d_grad_pre_activations, input_batch_saved, d_grad_W, M_in_q, N_out_q, batch_size
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Accumulate gradients for biases B (dL/dB)
    //    dL/dB_j = sum_batch (dL/dS_j)
    dim3 block_dim_grad_b(256);
    dim3 grid_dim_grad_b((N_out_q + block_dim_grad_b.x -1)/block_dim_grad_b.x, 1, 1);
    accumulate_grad_bias_kernel<<<grid_dim_grad_b, block_dim_grad_b>>>(
        d_grad_pre_activations, d_grad_b, N_out_q, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_grad_pre_activations);
}

void QuaternionDenseLayer::update_weights(float learning_rate) {
    // Simple SGD update: W = W - lr * dL/dW
    // This kernel can be a generic quaternion array update: A = A - lr * B
    int num_W_elements = N_out_q * M_in_q;
    int num_b_elements = N_out_q;

    // Simple way: loop and update. For large layers, a kernel is better.
    // For W
    std::vector<Quaternion> h_W(num_W_elements);
    std::vector<Quaternion> h_grad_W(num_W_elements);
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W, num_W_elements * sizeof(Quaternion), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_W.data(), d_grad_W, num_W_elements * sizeof(Quaternion), cudaMemcpyDeviceToHost));
    for(int i=0; i<num_W_elements; ++i) {
        h_W[i].w -= learning_rate * h_grad_W[i].w;
        h_W[i].x -= learning_rate * h_grad_W[i].x;
        h_W[i].y -= learning_rate * h_grad_W[i].y;
        h_W[i].z -= learning_rate * h_grad_W[i].z;
    }
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), num_W_elements * sizeof(Quaternion), cudaMemcpyHostToDevice));

    // For b
    std::vector<Quaternion> h_b(num_b_elements);
    std::vector<Quaternion> h_grad_b(num_b_elements);
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, num_b_elements * sizeof(Quaternion), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_b.data(), d_grad_b, num_b_elements * sizeof(Quaternion), cudaMemcpyDeviceToHost));
     for(int i=0; i<num_b_elements; ++i) {
        h_b[i].w -= learning_rate * h_grad_b[i].w;
        h_b[i].x -= learning_rate * h_grad_b[i].x;
        h_b[i].y -= learning_rate * h_grad_b[i].y;
        h_b[i].z -= learning_rate * h_grad_b[i].z;
    }
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), num_b_elements * sizeof(Quaternion), cudaMemcpyHostToDevice));
}

void QuaternionDenseLayer::zero_gradients() {
    CUDA_CHECK(cudaMemset(d_grad_W, 0, N_out_q * M_in_q * sizeof(Quaternion)));
    CUDA_CHECK(cudaMemset(d_grad_b, 0, N_out_q * sizeof(Quaternion)));
}