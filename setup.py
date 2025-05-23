from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Determine CUDA home robustly
cuda_home = os.environ.get('CUDA_HOME')
if not cuda_home:
    try:
        nvcc_path = os.popen('which nvcc').read().strip()
        if nvcc_path:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    except Exception:
        nvcc_path = None # Handle potential errors from os.popen

    if not cuda_home and os.path.exists('/usr/local/cuda'):
        cuda_home = '/usr/local/cuda'
    
    if not cuda_home:
        raise EnvironmentError(
            "CUDA_HOME environment variable is not set, nvcc was not found in PATH, "
            "and /usr/local/cuda does not exist. Please set CUDA_HOME or ensure nvcc is in your PATH."
        )

print(f"--- Using CUDA_HOME: {cuda_home}")
cuda_include_dir = os.path.join(cuda_home, 'include')
cuda_lib_dir = os.path.join(cuda_home, 'lib64') # Common for Linux; adjust if needed (e.g. 'lib' on some systems)

# Check if paths exist
if not os.path.exists(cuda_include_dir):
    raise FileNotFoundError(f"CUDA include directory not found: {cuda_include_dir}")
if not os.path.exists(cuda_lib_dir):
    # Try common alternatives for lib directory
    cuda_lib_dir_alt = os.path.join(cuda_home, 'lib')
    if os.path.exists(cuda_lib_dir_alt):
        cuda_lib_dir = cuda_lib_dir_alt
    else:
        cuda_lib_dir_alt_arch = os.path.join(cuda_home, 'lib', 'x86_64-linux-gnu') # For some installations
        if os.path.exists(cuda_lib_dir_alt_arch):
            cuda_lib_dir = cuda_lib_dir_alt_arch
        else:
            raise FileNotFoundError(f"CUDA library directory not found: {cuda_lib_dir} (and alternatives)")
print(f"--- Using CUDA Include directory: {cuda_include_dir}")
print(f"--- Using CUDA Library directory: {cuda_lib_dir}")


setup(
    name='quatnet_cuda',
    version='0.1.1', # Incremented version
    author='Commotum',
    author_email='your_email@example.com', # Replace with actual email
    description='CUDA Kernels for Isokawa Quaternion Layers in PyTorch',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    # packages=find_packages(where='python'), # Use if you have a 'python' subdirectory for Python modules
    # package_dir={'': 'python'},
    ext_modules=[
        CUDAExtension(
            name='quatnet_cuda', # This is how you'll import it: `import quatnet_cuda`
            sources=[
                'src/bindings.cpp',
                'src/isokawa_layer.cu',
                'src/quat_ops.cu', 
            ],
            include_dirs=[
                cuda_include_dir,
                os.path.abspath('src/') 
            ],
            library_dirs=[cuda_lib_dir],
            libraries=['cudart'], # CUDA Runtime library
            extra_compile_args={
                'cxx': [
                    '-g',                           # Debug symbols
                    '-std=c++17',                   # C++ standard
                    '-Wall',                        # Enable common warnings
                    '-Wextra',                      # Enable more warnings
                    '-Wno-unused-parameter',        # Suppress unused param warnings (common in stubs)
                    '-fPIC'                         # Position Independent Code for shared libraries
                ],
                'nvcc': [
                    '-O3',                          # Optimization level for CUDA code
                    '-std=c++17',                   # C++ standard for device code
                    '-gencode=arch=compute_86,code=sm_86', # For A6000 (Ampere)
                    # Add other architectures as needed, e.g.:
                    # '-gencode=arch=compute_75,code=sm_75', # For Turing
                    # '-gencode=arch=compute_70,code=sm_70', # For Volta
                    '--expt-relaxed-constexpr',     # Allow more flexible constexpr for some C++17 features
                    '-Xcompiler', '-Wall',          # Pass Wall to host compiler via nvcc
                    '-Xcompiler', '-Wextra',        # Pass Wextra to host compiler via nvcc
                    '-Xcompiler', '-Wno-unused-parameter',
                    '-Xcompiler', '-fPIC',          # Ensure host compiler also generates PIC
                    # '--threads', '0'  # Potentially speeds up compilation, uses all available cores
                ]
            }
        )
    ],
    cmdclass={
        # use_ninja=True can significantly speed up builds if Ninja is installed
        'build_ext': BuildExtension.with_options(use_ninja=os.environ.get('USE_NINJA', "0") == "1") 
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9', # Or your specific PyTorch version
    ],
)
