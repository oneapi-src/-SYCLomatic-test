#include <cuda.h>
#include <iostream>

void checkCUDAError(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        std::cerr << "CUDA Error: " << errorStr << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    CUcontext ctx1, ctx2;
    CUdevice device;
    CUresult result;

    // Initialize the CUDA Driver API
    result = cuInit(0);
    checkCUDAError(result);

    // Get the device
    result = cuDeviceGet(&device, 0);
    checkCUDAError(result);

    // Create the first context
    result = cuCtxCreate(&ctx1, 0, device);
    checkCUDAError(result);

    // Create the second context
    result = cuCtxCreate(&ctx2, 0, device);
    checkCUDAError(result);

    // Get the current context and push it onto the stack
    CUcontext currentCtx;
    result = cuCtxGetCurrent(&currentCtx);
    checkCUDAError(result);

    result = cuCtxPushCurrent(ctx1);
    checkCUDAError(result);

    // Now the current context is ctx1
    std::cout << "Context 1 is now current" << std::endl;

    // Push the current context (ctx1) and switch to ctx2
    result = cuCtxPushCurrent(ctx2);
    checkCUDAError(result);

    // Now the current context is ctx2
    std::cout << "Context 2 is now current" << std::endl;

    // Pop the context stack to switch back to ctx1
    result = cuCtxPopCurrent(&currentCtx);
    checkCUDAError(result);

    // currentCtx should be ctx1 now
    std::cout << "Context 1 is back to current" << std::endl;

    // Pop the context stack to switch back to the original context
    result = cuCtxPopCurrent(&currentCtx);
    checkCUDAError(result);

    // currentCtx should be the original context now
    std::cout << "Original context is back to current" << std::endl;

    // Cleanup
    cuCtxDestroy(ctx1);
    cuCtxDestroy(ctx2);

    return 0;
}
