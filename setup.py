# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from setuptools import setup
import os
import torch
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from packaging import version


include_dirs = os.path.dirname(os.path.abspath(__file__))
source_file = glob.glob(os.path.join('./', 'cuda_extension', '*.cpp'))
source_file += glob.glob(os.path.join('./', 'cuda_extension', '*.cu'))

if torch.cuda.is_available():
    if version.parse(torch.version.cuda) >= version.parse('10.2'):
        nvcc_arg = '--extended-lambda'
    else:
        # for cuda 10.1
        nvcc_arg = '--expt-extended-lambda'
    setup(
        name='r2d2lib',
        ext_modules=[
            CUDAExtension('r2d2lib',
                            sources=source_file,
                            include_dirs=[include_dirs],
                            extra_compile_args={'nvcc': [nvcc_arg],
                                                'cxx': ['-g']}
                            )
        ],
        cmdclass={'build_ext': BuildExtension})