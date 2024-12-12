//===--- peer_access_driver_api_tests.cu -------*- CUDA -*---===//
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
    CUdevice device1, device2;
    CUcontext context1, context2;

    // Initialize CUDA Driver API
    CUresult result = cuInit(0);

    cuDeviceGet(&device1, 0); // Device 0
    cuDeviceGet(&device2, 1); // Device 1

    // Create contexts for the devices
    cuCtxCreate(&context1, 0, device1);
    cuCtxCreate(&context2, 0, device2);

    // Enable peer access between the two contexts
    cuCtxSetCurrent(context1);
    result = cuCtxEnablePeerAccess(context2, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to enable peer access from device 0 to device 1\n";
    }

    int accessEnabled;
    result = cuDeviceCanAccessPeer(&accessEnabled, device1, device2);

    std::cout << "Peer-to-peer device access enabled.\n";

    return 0;
}

