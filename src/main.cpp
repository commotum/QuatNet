#include "quatnet_layer.h"
#include <iostream>

int main() {
    // Create layer: 4 inputs, 2 outputs
    QuaternionDenseLayer layer(4, 2);

    // Allocate input/output on GPU
    Quaternion *d_input, *d_output;
    cudaMalloc(&d_input, 4 * sizeof(Quaternion));
    cudaMalloc(&d_output, 2 * sizeof(Quaternion));

    // Host input example: all 1's
    Quaternion h_input[4] = {{1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1}};
    cudaMemcpy(d_input, h_input, 4 * sizeof(Quaternion), cudaMemcpyHostToDevice);

    // Forward pass
    layer.forward(d_input, d_output);

    // Fetch results back to host
    Quaternion h_output[2];
    cudaMemcpy(h_output, d_output, 2 * sizeof(Quaternion), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < 2; i++) {
        std::cout << "Output " << i << ": "
                  << h_output[i].w << ", "
                  << h_output[i].x << ", "
                  << h_output[i].y << ", "
                  << h_output[i].z << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
