#include "quatnet_layer.h"
#include <iostream>

int main() {
    QuaternionDenseLayer layer(1024, 512);
    Quaternion *d_input, *d_output;
    cudaMalloc(&d_input, 1024 * sizeof(Quaternion));
    cudaMalloc(&d_output, 512 * sizeof(Quaternion));
    Quaternion h_input[1024];
    for (int i = 0; i < 1024; i++) {
        h_input[i] = {1, 1, 1, 1};
    }
    cudaMemcpy(d_input, h_input, 1024 * sizeof(Quaternion), cudaMemcpyHostToDevice);
    for (int i = 0; i < 10000; i++) {
        layer.forward(d_input, d_output);
    }
    Quaternion h_output[512];
    cudaMemcpy(h_output, d_output, 512 * sizeof(Quaternion), cudaMemcpyDeviceToHost);
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
