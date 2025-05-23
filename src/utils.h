// src/utils.h
#pragma once
#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
}

// Define M_PI if not available (e.g. on MSVC without _USE_MATH_DEFINES)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif