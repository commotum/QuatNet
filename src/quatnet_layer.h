// quatnet_layer.h
#pragma once
#include "hamprod_kernel.h"

class QuatnetDenseLayer {
private:
    Quaternion* d_W;
    Quaternion* d_b;
    int N, M;
public:
    QuatnetDenseLayer(int input_dim, int output_dim);
    ~QuatnetDenseLayer();
    void forward(const Quaternion* d_input, Quaternion* d_output);
};