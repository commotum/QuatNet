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
cuda_lib_dir = os.path.join(cuda_home, 'lib64') 
if not os.path.exists(cuda_lib_dir): 
    cuda_lib_dir_alt = os.path.join(cuda_home, 'lib')
    if os.path.exists(cuda_lib_dir_alt):
        cuda_lib_dir = cuda_lib_dir_alt
    else:
        cuda_lib_dir_alt_arch = os.path.join(cuda_home, 'lib', 'x86_64-linux-gnu') 
        if os.path.exists(cuda_lib_dir_alt_arch):
            cuda_lib_dir = cuda_lib_dir_alt_arch
        else:
            raise FileNotFoundError(f"CUDA library directory not found: {cuda_lib_dir} (and alternatives)")
print(f"--- Using CUDA Include directory: {cuda_include_dir}")
print(f"--- Using CUDA Library directory: {cuda_lib_dir}")


setup(
    name='quatnet_cuda', 
    version='0.1.2', # Incremented version
    author='Commotum',
    author_email='your_email@example.com', 
    description='CUDA Kernels for Isokawa Quaternion Layers in PyTorch',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=[
        CUDAExtension(
            name='quatnet_cuda', 
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
            libraries=['cudart'], 
            extra_compile_args={
                'cxx': [ # Flags for GCC/Clang when compiling .cpp files (like bindings.cpp)
                    '-g',                         
                    '-std=c++17',                 
                    '-Wall',                      
                    '-Wextra',                    
                    '-Wno-unused-parameter',      
                    '-fPIC'                       
                ],
                'nvcc': [ # Flags for NVCC when compiling .cu files
                    '-O3',                        
                    '-std=c++17',                 
                    '-gencode=arch=compute_86,code=sm_86', 
                    '--expt-relaxed-constexpr',   
                    # Flags to be passed by NVCC to the host compiler (GCC/Clang)
                    '-Xcompiler', '-fPIC',
                    '-Xcompiler', '-Wall',        
                    '-Xcompiler', '-Wextra',      
                    '-Xcompiler', '-Wno-unused-parameter'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=os.environ.get('USE_NINJA', "0") == "1") 
    },
    python_requires='>=3.8',
    install_requires=['torch>=1.9'],
)