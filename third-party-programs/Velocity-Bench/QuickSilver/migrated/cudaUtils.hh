/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef HIPUTILS_HH
#define HIPUTILS_HH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

#if defined(HAVE_CUDA) || defined(HAVE_OPENMP_TARGET)
#endif

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)
#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(int err, const char *file, const int line)
{
}
inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
    #if defined (HAVE_CUDA)
        /*
        DPCT1010:34: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        dpct::err0 err = 0;

#endif
}

#ifdef HAVE_OPENMP_TARGET
#ifdef USE_OPENMP_NO_GPU
#define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#else
#define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#define HAVE_UVM
#endif
#elif HAVE_CUDA
#define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#define HAVE_UVM
#else
#define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif

enum ExecutionPolicy
{
    cpu,
    gpuWithCUDA,
    gpuWithOpenMP
};

inline ExecutionPolicy getExecutionPolicy(int useGPU)
{
    ExecutionPolicy execPolicy = ExecutionPolicy::cpu;

    if (useGPU)
    {
#if defined(HAVE_CUDA)
        execPolicy = ExecutionPolicy::gpuWithCUDA;
#elif defined(HAVE_OPENMP_TARGET)
        execPolicy = ExecutionPolicy::gpuWithOpenMP;
#endif
    }
    return execPolicy;
}

template <class T>
inline void gpuMallocManaged(T **ptr, size_t size, unsigned int flags = 0)
{
     #if defined (HAVE_CUDA)
        //   cudaMallocManaged(ptr,size);
          #ifdef UNIFIED_HOST                    
                cudaMallocHost(ptr, size);
          #elif defined(UNIFIED_DEVICE)                    
                cudaMalloc(ptr, size);
          #else
                /*
                DPCT1064:36: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                safeCall(
                    DPCT_CHECK_ERROR(*ptr = (T *)sycl::malloc_shared(
                                         size, dpct::get_in_order_queue())));
#endif
     #elif defined (HAVE_HIP)
       #ifdef UNIFIED_HOST
          hipHostMalloc(ptr,size,flags);
       #elif defined(UNIFIED_DEVICE)
          hipMalloc(ptr,size);
       #else
          hipMallocManaged(ptr,size);
       #endif
     #endif

}

template <class T>
inline void gpuFree(T *ptr)
{
     #if defined (HAVE_CUDA)
           safeCall(DPCT_CHECK_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
#elif defined (HAVE_HIP)
          hipHostFree(ptr);
     #endif
}
#endif
