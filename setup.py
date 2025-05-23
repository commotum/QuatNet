from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_home = os.environ.get('CUDA_HOME')
if not cuda_home:
    try:
        nvcc_path = os.popen('which nvcc').read().strip()
        if nvcc_path:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    except Exception:
        nvcc_path = None
    if not cuda_home and os.path.exists('/usr/local/cuda'):
        cuda_home = '/usr/local/cuda'
    if not cuda_home:
        raise EnvironmentError("CUDA_HOME not set and nvcc not found.")

print(f"--- Using CUDA_HOME: {cuda_home}")
cuda_include_dir = os.path.join(cuda_home, 'include')
cuda_lib_dir = os.path.join(cuda_home, 'lib64')
if not os.path.exists(cuda_lib_dir): # Try 'lib' if 'lib64' doesn't exist
    cuda_lib_dir_alt = os.path.join(cuda_home, 'lib')
    if os.path.exists(cuda_lib_dir_alt):
        cuda_lib_dir = cuda_lib_dir_alt
    else:
        raise FileNotFoundError(f"CUDA library directory not found: {cuda_lib_dir} or {cuda_lib_dir_alt}")


setup(
    name='quatnet_cuda',
    version='0.2.0', # Version bump
    author='Commotum',
    description='CUDA Kernels for Quaternion Neural Networks in PyTorch',
    ext_modules=[
        CUDAExtension(
            name='quatnet_cuda',
            sources=[
                'src/bindings.cpp',         # Pybind11 C++ to Python interface
                'src/quatnet_layer.cu',     # Your C++ QuaternionDenseLayer (Parcollet-style)
                'src/hamprod_kernel.cu',    # Used by quatnet_layer.cu
                'src/quat_ops.cu',          # For Quaternion struct and basic __device__ ops
                # Do NOT include isokawa_layer.cu if we are not using the rotational layer here
            ],
            include_dirs=[
                cuda_include_dir,
                os.path.abspath('src/')
            ],
            library_dirs=[cuda_lib_dir],
            libraries=['cudart', 'curand'], # Added curand if your quatnet_layer.cpp uses it
            extra_compile_args={
                'cxx': ['-g', '-std=c++17', '-Wall', '-Wextra', '-Wno-unused-parameter', '-fPIC'],
                'nvcc': [
                    '-O3', '-std=c++17',
                    '-gencode=arch=compute_86,code=sm_86', # For A6000
                    '--expt-relaxed-constexpr',
                    '-Xcompiler', '-Wall', '-Wextra', '-Wno-unused-parameter', '-fPIC'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=os.environ.get('USE_NINJA', "0") == "1")},
    python_requires='>=3.8',
    install_requires=['torch>=1.9'],
)
