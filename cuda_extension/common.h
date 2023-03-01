// coding=utf-8
// Copyright (c) 2022 Ant Group
// Author: Xiang Hu

#ifndef R2D2_COMMON_H
#define R2D2_COMMON_H
#include <stdio.h>
#include <c10/cuda/CUDACachingAllocator.h>

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(fn_call)                          \
    if (fn_call != cudaSuccess)                        \
    {                                                  \
        c10::cuda::CUDACachingAllocator::emptyCache(); \
        HandleError(fn_call, __FILE__, __LINE__);      \
    }

#endif