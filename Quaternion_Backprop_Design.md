Comparative Analysis of Quaternion Backpropagation and Implementation Strategy for Image Compression
-----------------------------------------------------------------------------------------------

This document outlines two approaches to quaternion neural networks (QNNs) and how QuatNet will adopt a performant design for image compression experiments.

### 1. Isokawa et al. QNN
* **Forward pass**
  * Uses $|w| w x \bar w$ for each connection, followed by a purely imaginary sigmoid activation.
* **Backward pass**
  * Gradients require several Hamilton products and explicit derivatives of the rotational formula, making the method computationally expensive.
* **Use case**
  * Geometric transformations such as those used in the original image compression paper.

### 2. Parcollet et al. Style QNN / QRNN
* **Forward pass**
  * Based purely on Hamilton products: $s_j = W_j \otimes x + b_j$ with split activations.
* **Backward pass**
  * Gradients have concise quaternion expressions, e.g. $\nabla_W E = \delta \otimes h^*$ where $^*$ denotes quaternion conjugation.
* **Advantages**
  * Fewer operations per weight, easier to optimize with CUDA kernels such as the provided `hamprod_kernel.cu`.

### 3. Implementation Decision
QuatNet will implement Isokawa's image compression autoencoder using the Parcollet-style operations. Inputs remain purely imaginary quaternions representing RGB pixels, but layers use standard Hamilton-product based dense layers (`QuaternionDenseLayer`).

The decoder output applies a "pure imaginary" sigmoid so that the real part remains zero while RGB components are in $[0,1]$. The mean squared error loss ignores the real component.

### 4. Benefits
* **Performance** – relies on the optimized Hamilton product kernel.
* **Simplicity** – one forward and backward formulation for all dense QNN layers.
* **Extensibility** – the same layer can be reused for RNNs or other architectures.

This strategy keeps the geometric interpretation of quaternion inputs for image compression while leveraging a modern, efficient quaternion backpropagation scheme.
