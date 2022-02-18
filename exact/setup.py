from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='exact',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'exact.cpp_extension.calc_precision',
              ['exact/cpp_extension/calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'exact.cpp_extension.backward_func',
              ['exact/cpp_extension/backward_func.cc', 'exact/cpp_extension/backward_func_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'exact.cpp_extension.quantization',
              ['exact/cpp_extension/quantization.cc', 'exact/cpp_extension/quantization_cuda_kernel.cu'],
              extra_compile_args={'nvcc': ['--expt-extended-lambda']}
          ),
          cpp_extension.CUDAExtension(
              'exact.cpp_extension.spmm',
              ['exact/cpp_extension/spmm.cc', 'exact/cpp_extension/spmm_cuda.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
