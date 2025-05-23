// src/isokawa_layer.cu
#include "isokawa_layer.h"
#include "quat_ops.h"  // Add explicit include for Quaternion operations
#include "utils.h" // For CUDA_CHECK and M_PI
#include <vector>         // For host-side initialization arrays
#include <cstdlib>        // For rand, srand
#include <cmath>          // For logf, cosf, sqrtf on host for init

// Kernel for Isokawa rotational neuron computation (s_j part)
__global__ void isokawaRotationalForwardKernel(
    const Quaternion* __restrict__ W,
    const Quaternion* __restrict__ X_batch,
    const Quaternion* __restrict__ theta,
    Quaternion* __restrict__ S_internal_batch,
    int M_in, int N_out, int batch_size) 
{
    int batch_idx = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }

    Quaternion s_j_acc = {0.f, 0.f, 0.f, 0.f};
    const Quaternion* X_current_sample = X_batch + batch_idx * M_in;

    for (int i = 0; i < M_in; ++i) {
        Quaternion w_ji = W[j * M_in + i];
        Quaternion x_i = X_current_sample[i]; // x_i.w is assumed 0 (pure input)

        float w_norm_val = w_ji.norm(); // Uses the stabilized norm from quat_ops.h
        Quaternion u_ji;
        // Check against a slightly larger epsilon than in norm() itself, 
        // as norm() already added epsilon for sqrtf.
        if (w_norm_val < 1e-6f) { 
            u_ji = {0.f, 0.f, 0.f, 0.f}; 
        } else {
            u_ji = w_ji.scale(1.0f / w_norm_val);
        }
        
        // Rotation: u_ji * x_i * u_ji.conjugate()
        // Since x_i.w is 0, the result will also have w approx 0.
        s_j_acc = s_j_acc + (u_ji * x_i * u_ji.conjugate());
    }
    
    // theta[j].w is assumed 0 (pure threshold)
    s_j_acc = s_j_acc - theta[j]; 
    S_internal_batch[batch_idx * N_out + j] = s_j_acc;
}

// Kernel for Isokawa's activation function
__device__ float device_sigmoid(float val) {
    return 1.0f / (1.0f + expf(-val));
}

__global__ void isokawaActivationKernel(
    const Quaternion* __restrict__ S_internal_batch,
    Quaternion* __restrict__ Y_batch,
    int N_out, int batch_size)
{
    int batch_idx = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || j >= N_out) {
        return;
    }

    int global_idx = batch_idx * N_out + j;
    Quaternion s_j = S_internal_batch[global_idx];

    // Output is pure quaternion, s_j.w should be near zero.
    Y_batch[global_idx] = {
        0.0f, 
        device_sigmoid(s_j.x),
        device_sigmoid(s_j.y),
        device_sigmoid(s_j.z)
    };
}

// --- IsokawaQuaternionLayer Class Implementation ---
void IsokawaQuaternionLayer::initializeParameters() {
    float fan_in_float = static_cast<float>(M_in);
    float fan_out_float = static_cast<float>(N_out);
    
    float sigma_W_sq = 2.0f / (fan_in_float + fan_out_float);
    if (fan_in_float + fan_out_float == 0) sigma_W_sq = 0.1f; // Avoid div by zero for 0-dim layers
    float sigma_W = std::sqrt(sigma_W_sq);

    std::vector<Quaternion> h_W(N_out * M_in);
    srand(static_cast<unsigned int>(time(0))); 

    for (int i = 0; i < N_out * M_in; ++i) {
        // Simple uniform init for components, then normalize on GPU, or use Box-Muller for normal
        // For this example, let's make weights that result in norms around 1 after init.
        // Or initialize components so that the expected norm is reasonable.
        // Host-side Box-Muller for each component:
        float r1 = static_cast<float>(rand()) / RAND_MAX;
        float r2 = static_cast<float>(rand()) / RAND_MAX;
        h_W[i].w = sigma_W * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
        r1 = static_cast<float>(rand()) / RAND_MAX; r2 = static_cast<float>(rand()) / RAND_MAX;
        h_W[i].x = sigma_W * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
        r1 = static_cast<float>(rand()) / RAND_MAX; r2 = static_cast<float>(rand()) / RAND_MAX;
        h_W[i].y = sigma_W * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
        r1 = static_cast<float>(rand()) / RAND_MAX; r2 = static_cast<float>(rand()) / RAND_MAX;
        h_W[i].z = sigma_W * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
    }
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(Quaternion), cudaMemcpyHostToDevice));

    // Initialize thresholds (pure quaternions, small random imaginary parts)
    std::vector<Quaternion> h_theta(N_out);
    float sigma_theta = 0.01f; 
    for (int i = 0; i < N_out; ++i) {
        h_theta[i].w = 0.0f; // Pure quaternion
        float r1 = static_cast<float>(rand()) / RAND_MAX; float r2 = static_cast<float>(rand()) / RAND_MAX;
        h_theta[i].x = sigma_theta * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
        r1 = static_cast<float>(rand()) / RAND_MAX; r2 = static_cast<float>(rand()) / RAND_MAX;
        h_theta[i].y = sigma_theta * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
        r1 = static_cast<float>(rand()) / RAND_MAX; r2 = static_cast<float>(rand()) / RAND_MAX;
        h_theta[i].z = sigma_theta * std::sqrt(-2.0f * std::log(r1 + 1e-9f)) * std::cos(2.0f * M_PI * r2);
    }
    CUDA_CHECK(cudaMemcpy(d_theta, h_theta.data(), h_theta.size() * sizeof(Quaternion), cudaMemcpyHostToDevice));
}

IsokawaQuaternionLayer::IsokawaQuaternionLayer(int input_dim, int output_dim)
    : M_in(input_dim), N_out(output_dim) {
    if (M_in <=0 || N_out <=0) {
        throw std::invalid_argument("Input and output dimensions must be positive.");
    }
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(N_out) * M_in * sizeof(Quaternion)));
    CUDA_CHECK(cudaMalloc(&d_theta, N_out * sizeof(Quaternion)));
    initializeParameters();
}

IsokawaQuaternionLayer::~IsokawaQuaternionLayer() {
    cudaFree(d_W);
    cudaFree(d_theta);
}

void IsokawaQuaternionLayer::forward(const Quaternion* d_batch_X, 
                                     Quaternion* d_batch_Y, 
                                     int batch_size,
                                     Quaternion* d_pre_activation_S_optional) {
    if (batch_size <=0) {
        throw std::invalid_argument("Batch size must be positive.");
    }
    
    dim3 blockDim(256); 
    dim3 gridDimRotational;
    gridDimRotational.x = (N_out + blockDim.x - 1) / blockDim.x;
    gridDimRotational.y = batch_size;
    gridDimRotational.z = 1;

    Quaternion* s_buffer_ptr;
    bool allocated_s_buffer = false;
    if (d_pre_activation_S_optional) {
        s_buffer_ptr = d_pre_activation_S_optional;
    } else {
        CUDA_CHECK(cudaMalloc(&s_buffer_ptr, static_cast<size_t>(batch_size) * N_out * sizeof(Quaternion)));
        allocated_s_buffer = true;
    }

    isokawaRotationalForwardKernel<<<gridDimRotational, blockDim>>>(
        d_W, d_batch_X, d_theta, s_buffer_ptr, M_in, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError()); 

    isokawaActivationKernel<<<gridDimRotational, blockDim>>>( // Same grid/block for activation
        s_buffer_ptr, d_batch_Y, N_out, batch_size);
    CUDA_CHECK(cudaGetLastError());

    if (allocated_s_buffer) {
        cudaFree(s_buffer_ptr);
    }
}