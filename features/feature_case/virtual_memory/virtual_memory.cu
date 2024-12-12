// ====------ virtual_memory.cu--------------- *- CUDA -*---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <iostream>

#define SIZE 100

int main() {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t granularity;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);   
    size_t POOL_SIZE =  granularity;

    CUdeviceptr reserved_addr;
    CUmemGenericAllocationHandle allocHandle;
    cuMemAddressReserve(&reserved_addr, POOL_SIZE, 0, 0, 0);
    cuMemCreate(&allocHandle, POOL_SIZE, &prop, 0);
    cuMemMap(reserved_addr, POOL_SIZE, 0, allocHandle, 0);

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(reserved_addr, POOL_SIZE, &accessDesc, 1);
    int* host_data = new int[SIZE];
    int* host_data2 = new int[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        host_data[i] = i;
        host_data2[i] = 0;
    }

    cuMemcpyHtoD(reserved_addr, host_data, SIZE * sizeof(int));
    cuMemcpyDtoH(host_data2, reserved_addr, SIZE * sizeof(int));

    for (int i = 0; i < SIZE; ++i) {
        if(host_data[i] != host_data2[i]) {
          std::cout << "test failed" << std::endl;
          exit(-1);
        }
    }
    std::cout << "test passed" << std::endl;

    cuMemUnmap(reserved_addr, POOL_SIZE);
    cuMemRelease(allocHandle);
    cuMemAddressFree(reserved_addr, POOL_SIZE);

    delete[] host_data;
    delete[] host_data2;

    cuCtxDestroy(context);
    return 0;
}
