from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Determine CUDA home
cuda_home = os.environ.get('CUDA_HOME')
if not cuda_home:
    nvcc_path = os.popen('which nvcc').read().strip()
    if nvcc_path:
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    elif os.path.exists('/usr/local/cuda'):
        cuda_home = '/usr/local/cuda'
    else:
        raise EnvironmentError("CUDA_HOME not set and nvcc not found.")

print(f"Using CUDA_HOME: {cuda_home}")

setup(
    name='quatnet_cuda',
    version='0.1.0',
    author='Your Name / Commotum',
    author_email='your_email@example.com',
    description='CUDA Kernels for Isokawa Quaternion Layers in PyTorch',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=[
        CUDAExtension(
            name='quatnet_cuda',
            sources=[
                'src/bindings.cpp',
                'src/isokawa_layer.cu',
                'src/quat_ops.cu',
            ],
            include_dirs=[
                os.path.join(cuda_home, 'include'),
                os.path.abspath('src/')
            ],
            library_dirs=[
                os.path.join(cuda_home, 'lib64'),
            ],
            libraries=['cudart'],
            extra_compile_args={
                'cxx': ['-g', '-std=c++17', '-Wall', '-Wextra'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-gencode=arch=compute_86,code=sm_86',
                    '--expt-relaxed-constexpr',
                    '-Xcompiler', '-Wall', '-Wextra', '-Wno-unused-parameter'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9',
    ],
)
