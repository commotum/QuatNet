from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quatnet_cuda',
    ext_modules=[
        CUDAExtension('quatnet_cuda', [
            'src/isokawa_layer.cu',
            'src/quat_ops.cu',
            'src/hamprod_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 