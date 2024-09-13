//===------------- context_push_n_pop.cu -------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===------------------------------------------------------ -===//
#include <cuda.h>
#include <iostream>


int main() {
    CUcontext ctx1, ctx2;
    CUdevice device;

    // Initialize the CUDA Driver API
    cuInit(0);

    // Get the device
    cuDeviceGet(&device, 0);

    // Create the first context
    cuCtxCreate(&ctx1, 0, device);

    // Create the second context
    cuCtxCreate(&ctx2, 0, device);

    // Get the current context and push it onto the stack
    CUcontext currentCtx;
    cuCtxGetCurrent(&currentCtx);

    cuCtxPushCurrent(ctx1);

    // Now the current context is ctx1
    std::cout << "Context 1 is now current" << std::endl;

    // Push the current context (ctx1) and switch to ctx2
    cuCtxPushCurrent(ctx2);

    // Now the current context is ctx2
    std::cout << "Context 2 is now current" << std::endl;

    // Pop the context stack to switch back to ctx1
    cuCtxPopCurrent(&currentCtx);

    // currentCtx should be ctx1 now
    std::cout << "Context 1 is back to current" << std::endl;

    // Pop the context stack to switch back to the original context
    cuCtxPopCurrent(&currentCtx);

    // currentCtx should be the original context now
    std::cout << "Original context is back to current" << std::endl;

    // Cleanup
    cuCtxDestroy(ctx1);
    cuCtxDestroy(ctx2);

    return 0;
}
