# QuatNet
A library for Quaternion Neural Networks, powered by the HamProd Kernel. Accelerate deep learning with fewer parameters using efficient quaternion operations on Nvidia GPUs. Supports QNN layers and QRNNs for multidimensional data like images and 3D rotations.

## Overview
QuatNet is a framework for building Quaternion Neural Networks (QNNs), which use Quaternions (four-dimensional hypercomplex numbers that contain a real and three separate imaginary components) to process multidimensional data efficiently. Inspired by the need for optimized Quaternion operations outlined in the ICLR 2019 paper [*Quaternion Recurrent Neural Networks*](https://arxiv.org/abs/1806.04418), QuatNet introduces the **HamProd Kernel**, a standalone CUDA kernel designed for high-performance Hamilton product computations on Nvidia GPUs. By leveraging Quaternions, QuatNet achieves up to 4x fewer parameters than real-valued networks, with reduced CPU-GPU memory transfers, making it ideal for tasks like image processing, 3D kinematics, and speech recognition.

## Features
- **HamProd Kernel**: A fast CUDA kernel for the Hamilton product, optimized for GPU efficiency.
- **QNN Layers**: Dense and recurrent layers for Quaternion-based neural networks.
- **Efficiency**: Fewer parameters and lower memory overhead compared to real-valued networks.
- **Applications**: Supports multidimensional data, including RGB images, 3D orientations, and sequential tasks via QRNNs.

## Installation
### Requirements
- Nvidia GPU with CUDA Toolkit 11.8 or later
- CMake 3.10+
- C++ compiler (e.g., g++)

### Build Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/QuatNet.git
   cd QuatNet
   ```
2. Create a build directory and compile:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the example:
   ```bash
   ./quatnet
   ```

## Usage
QuatNet provides a simple API for building QNNs. Below is an example of using a quaternion dense layer:

```cpp
#include "quatnet_layer.h"

int main() {
    // Initialize layer: 4 inputs, 2 outputs
    QuaternionDenseLayer layer(4, 2);
    
    // Allocate input/output on GPU
    Quaternion *d_input, *d_output;
    cudaMalloc(&d_input, 4 * sizeof(Quaternion));
    cudaMalloc(&d_output, 2 * sizeof(Quaternion));
    
    // Set input (example)
    Quaternion h_input[4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
    cudaMemcpy(d_input, h_input, 4 * sizeof(Quaternion), cudaMemcpyHostToDevice);
    
    // Run forward pass
    layer.forward(d_input, d_output);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

See `src/main.cpp` for a complete example.

## Motivation
QuatNet was inspired by the ICLR 2019 paper [*Quaternion Recurrent Neural Networks*](https://arxiv.org/abs/1806.04418), which noted that Quaternion neural networks, while computationally intensive due to the Hamilton product, could outperform real-valued networks with a properly engineered CUDA kernel. The **HamProd Kernel** addresses this by providing a high-performance, GPU-accelerated implementation, enabling QNNs to process data with fewer parameters and faster training/inference times.

## Project Structure
```
QuatNet/
├── src/
│   ├── hamprod_kernel.cu    # Hamilton Product Kernel
│   ├── hamprod_kernel.h
│   ├── quatnet_layer.cpp    # QNN layer implementation
│   ├── quatnet_layer.h
│   ├── main.cpp             # Example program
├── tests/                   # Unit tests (planned)
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── LICENSE                  # MIT License
```

## Contributing
Contributions are welcome! Whether you're adding new QNN layers, optimizing the **HamProd Kernel**, or testing on different GPUs, please:
1. Fork the repo.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to your fork (`git push origin feature/your-feature`).
5. Open a pull request.

## License
[MIT License](LICENSE)

## Acknowledgments
- Inspired by [*Quaternion Recurrent Neural Networks*](https://arxiv.org/abs/1806.04418) by Parcollet et al., ICLR 2019.
- Built with CUDA for Nvidia GPUs.

## Contact
For questions or ideas, open an issue or reach out to @commotum on X.