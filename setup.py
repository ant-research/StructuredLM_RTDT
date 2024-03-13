from setuptools import setup
import os
import torch
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


include_dirs = os.path.dirname(os.path.abspath(__file__))
source_file = glob.glob(os.path.join('./', 'cpp_extension', '*.cpp'))

if torch.cuda.is_available():
    setup(
        name='cppbackend',
        ext_modules=[
            CppExtension('cppbackend',
                          sources=source_file,
                          include_dirs=[include_dirs]
                         )
        ],
        cmdclass={'build_ext': BuildExtension})