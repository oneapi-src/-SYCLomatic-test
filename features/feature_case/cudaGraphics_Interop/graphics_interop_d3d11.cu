// ===-------- graphics_interop_d3d11.cu ------- *- CUDA -* ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>

// CUDA-DirectX11 interop header
#include <cuda_d3d11_interop.h>

// DirectX headers
#include <d3d11.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")


#define PRINT_PASS 1

#define CHECK_CUDA_ERROR(call, errMsg, stsMsg) \
    do { \
        try { \
            cudaError_t cu_status = call; \
            cudaDeviceSynchronize(); \
            cu_status = cudaGetLastError(); \
            if (cu_status != cudaSuccess) { \
                std::cout << "[ERROR] " << errMsg << std::endl; \
                std::cout << "[ERROR] " << cu_status << ": " << cudaGetErrorName(cu_status) << std::endl; \
            } else { \
                std::cout << "[SUCCESS] " << stsMsg << std::endl; \
            } \
        } \
        catch (const std::exception& e) {                                      \
            std::cerr << "[ERROR]: " << e.what() << std::endl;                 \
        }                                                                      \
    } while(0)

#define CHECK_D3D11_ERROR(call, errMsg, stsMsg) \
    do { \
        HRESULT d11_status = call; \
        if (d11_status != S_OK) { \
            std::cout << "[ERROR] " << errMsg << std::endl; \
        } else { \
            std::cout << "[SUCCESS] " << stsMsg << std::endl; \
        } \
    } while(0)


#define WIDTH 16
#define HEIGHT 16


// CUDA kernel for processing data
__global__ void updateTexData(float* data, int pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Perform some computation on the data
        int index = y * pitch + x;
        data[index] += 2.0f;
    }
}


ID3D11Device* create_d3d11_dev(ID3D11DeviceContext** d3dContext, IDXGIAdapter1* pAdapter1) {
    ID3D11Device* d3dDevice;

    CHECK_D3D11_ERROR(
        D3D11CreateDevice(
            pAdapter1, /*nullptr*/
            D3D_DRIVER_TYPE_UNKNOWN, /*D3D_DRIVER_TYPE_HARDWARE*/
            nullptr,
            0,
            nullptr,
            0,
            D3D11_SDK_VERSION,
            &d3dDevice,
            nullptr,
            d3dContext
        ),
        "Cannot create D3D11 device",
        "Created D3D11 device"
    );

    return d3dDevice;
}

ID3D11Texture2D* create_d3d11_tex(D3D11_TEXTURE2D_DESC &texDesc, ID3D11Device* d3dDevice) {
    ID3D11Texture2D *d3dTexture;

    ZeroMemory(&texDesc, sizeof(texDesc));
    texDesc.Width = WIDTH;
    texDesc.Height = HEIGHT;
    texDesc.ArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R32_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    texDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    CHECK_D3D11_ERROR(
        d3dDevice->CreateTexture2D(&texDesc, nullptr, &d3dTexture),
        "Cannot create D3D11 2D texture",
        "Created D3D11 2D texture"
    );

    return d3dTexture;
}

int main() {
    // Check CUDA-D3D11 env
    int cudaDevIx = -1;
    IDXGIFactory1* pFactory1 = nullptr;
    IDXGIAdapter1* pAdapter1 = nullptr;

    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory1))) {
        if (pFactory1->EnumAdapters1(1, &pAdapter1) == DXGI_ERROR_NOT_FOUND)
            return 1;
    }
    
    // Initialize DirectX
    ID3D11DeviceContext* d3dContext = nullptr;
    ID3D11Device* d3dDevice = create_d3d11_dev(&d3dContext, pAdapter1);

    // Create a texture for DirectX and CUDA interoperability
    D3D11_TEXTURE2D_DESC texDesc;
    ID3D11Texture2D* d3dTexture = create_d3d11_tex(texDesc, d3dDevice);

    // STEP: 1
    // Register the DirectX resource with CUDA
    cudaGraphicsResource_t cudaResource;
    CHECK_CUDA_ERROR(
        cudaGraphicsD3D11RegisterResource(&cudaResource, d3dTexture, cudaGraphicsRegisterFlagsNone),
        "Cannot register D3D11 2D texture with CUDA resource",
        "Registered D3D11 2D texture with CUDA resource"
    );

    // STEP: 2
    // Set the flags for CUDA resource mapping
    CHECK_CUDA_ERROR(
        cudaGraphicsResourceSetMapFlags(cudaResource, cudaGraphicsMapFlagsNone),
        "Cannot set map flags for CUDA resource",
        "Set map flags for CUDA resource"
    );

    // STEP: 3
    // Map the CUDA resource for access
    CHECK_CUDA_ERROR(
        cudaGraphicsMapResources(1, &cudaResource),
        "Cannot map CUDA resource",
        "Mapped CUDA resource"
    );

    // STEP: 4
    // Get the mapped array from the CUDA resource
    cudaArray_t cudaArr;
    CHECK_CUDA_ERROR(
        cudaGraphicsSubResourceGetMappedArray(&cudaArr, cudaResource, 0, 0),
        "Cannot aquire texture data as a CUDA array",
        "Aquired texture data as a CUDA array"
    );

    if (!cudaArr)
        std::cout << "[ERROR] cudaArr is nullptr" << std::endl;

    /*
    // STEP: 5
    // Test copying dummy data to the imported texture memory
    bool status = write_to_imported_tex_mem(cudaArr);
    std::cout << "[STATUS] Result: " << (status? "Success": "Fail") << std::endl;
    */

    // STEP: 6
    // Unmap the CUDA resource
    CHECK_CUDA_ERROR(
        cudaGraphicsUnmapResources(1, &cudaResource),
        "Cannot unmap CUDA resource",
        "Unmapped CUDA resource"
    );

    // STEP: 7
    // Unregister the CUDA resource
    CHECK_CUDA_ERROR(
        cudaGraphicsUnregisterResource(cudaResource),
        "Cannot unregister D3D11 2D texture with CUDA resource",
        "Unregistered D3D11 2D Texture with CUDA resource"
    );

    // Cleanup
    pFactory1->Release();
    pAdapter1->Release();

    d3dTexture->Release();
    d3dContext->Release();
    d3dDevice->Release();

    cudaDeviceReset();

    return 0;
}
