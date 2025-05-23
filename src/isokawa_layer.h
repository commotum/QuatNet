// src/isokawa_layer.h
#pragma once
#include "quat_ops.h" // For Quaternion struct
#include <curand.h>          // For weight initialization (if using cuRAND)

class IsokawaQuaternionLayer {
private:
    Quaternion* d_W;         // Weights: N_out x M_in
    Quaternion* d_theta;     // Thresholds: N_out (pure quaternions)
    
    int M_in;                // Input dimension (number of pure quaternions)
    int N_out;               // Output dimension (number of pure quaternions)

    // curandGenerator_t curand_gen; // Uncomment if using cuRAND for init

    void initializeParameters(); // Helper for constructor

public:
    IsokawaQuaternionLayer(int input_dim, int output_dim);
    ~IsokawaQuaternionLayer();

    void forward(const Quaternion* d_batch_X, 
                 Quaternion* d_batch_Y, 
                 int batch_size,
                 Quaternion* d_pre_activation_S_optional = nullptr);
    
    // Expose pointers for external management or gradient updates if needed
    Quaternion* getWeightsPtr() { return d_W; }
    Quaternion* getThresholdsPtr() { return d_theta; }
    int getInputDim() const { return M_in; }
    int getOutputDim() const { return N_out; }
};