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

#define CHECK_D3D11_ERROR(call, errMsg)                     \
    do                                                      \
    {                                                       \
        HRESULT d11_status = call;                          \
        if (d11_status != S_OK) {                           \
            std::cout << "[ERROR] " << errMsg << std::endl; \
        }                                                   \
    } while (0)

ID3D11Device *create_d3d11_dev(ID3D11DeviceContext **d3dContext, IDXGIAdapter1 *pAdapter1) {
    ID3D11Device *d3dDevice;

    CHECK_D3D11_ERROR(
        D3D11CreateDevice(
            pAdapter1,               /*nullptr*/
            D3D_DRIVER_TYPE_UNKNOWN, /*D3D_DRIVER_TYPE_HARDWARE*/
            nullptr,
            0,
            nullptr,
            0,
            D3D11_SDK_VERSION,
            &d3dDevice,
            nullptr,
            d3dContext),
        "Cannot create D3D11 device");

    return d3dDevice;
}

ID3D11Texture2D *create_d3d11_tex(ID3D11Device *d3dDevice, int w, int h, DXGI_FORMAT dxFormat = DXGI_FORMAT_R32_FLOAT,
                                  int aSize = 1, int mLevels = 1) {
    D3D11_TEXTURE2D_DESC texDesc;
    ID3D11Texture2D *d3dTexture;

    ZeroMemory(&texDesc, sizeof(texDesc));
    texDesc.Width = w;
    texDesc.Height = h;
    texDesc.ArraySize = aSize;
    texDesc.MipLevels = mLevels;
    texDesc.Format = dxFormat;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    texDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    CHECK_D3D11_ERROR(
        d3dDevice->CreateTexture2D(&texDesc, nullptr, &d3dTexture),
        "Cannot create D3D11 texture");

    return d3dTexture;
}

cudaTextureObject_t getTex(cudaArray_t input) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = input;

    cudaTextureDesc texDesc = {};

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    return tex;
}

template <typename T>
__global__ void accessTexture(T *output, cudaTextureObject_t texObj, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        // Access the texture element
        T value = tex2D<T>(texObj, x, y);
        output[y * width + x] = value;
    }
}

cudaGraphicsResource_t initInterop(ID3D11Resource *d3dResource) {
    cudaGraphicsResource_t cudaResource;

    // Register the DirectX resource with CUDA
    cudaGraphicsD3D11RegisterResource(&cudaResource, d3dResource, cudaGraphicsRegisterFlagsNone);

    // Set the flags for CUDA resource mapping
    cudaGraphicsResourceSetMapFlags(cudaResource, cudaGraphicsMapFlagsNone);

    // Map the CUDA resource for access
    cudaGraphicsMapResources(1, &cudaResource);

    return cudaResource;
}

void cleanupInterop(cudaGraphicsResource_t cudaResource) {
    // Unmap the CUDA resource
    cudaGraphicsUnmapResources(1, &cudaResource);

    // Unregister the CUDA resource
    cudaGraphicsUnregisterResource(cudaResource);
}

int main() {
    // Init DX env
    IDXGIFactory1 *pFactory1 = nullptr;
    IDXGIAdapter1 *pAdapter1 = nullptr;
    ID3D11DeviceContext *d3dContext = nullptr;
    ID3D11Device *d3dDevice = nullptr;

    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory1))) {
        for (UINT adapterIndex = 0; pFactory1->EnumAdapters1(adapterIndex, &pAdapter1) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
            DXGI_ADAPTER_DESC1 desc;
            pAdapter1->GetDesc1(&desc);

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                continue;
            }

            d3dDevice = create_d3d11_dev(&d3dContext, pAdapter1);

            if (d3dDevice != nullptr) {
                break;
            }
        }
    }

    {
        // Init test data
        const int w = 16;
        const int h = 16;

        float input[h * w] = {
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
        };
        float *output;
        cudaMallocManaged(&output, sizeof(input));

        // Create a texture for DirectX and CUDA interoperability
        ID3D11Texture2D *d3dTexture = create_d3d11_tex(d3dDevice, w, h);

        // Init interop
        cudaGraphicsResource_t cudaResource = initInterop(d3dTexture);
        
        // Get the mapped array from the CUDA resource
        cudaArray_t cudaArr;
        cudaGraphicsSubResourceGetMappedArray(&cudaArr, cudaResource, 0, 0);

        // Access the underlying memory of interop CUDA resource
        cudaMemcpy2DToArray(cudaArr, 0, 0, input, sizeof(float) * w,
                            sizeof(float) * w, h, cudaMemcpyHostToDevice);
        cudaTextureObject_t tex = getTex(cudaArr);

        dim3 blockSize(4, 4);
        dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
        accessTexture<<<gridSize, blockSize>>>(output, tex, w, h);
        cudaDeviceSynchronize();

        cleanupInterop(cudaResource);

        bool pass = true;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (output[i * w + j] != input[i * w + j]) {
                    std::cout << "Failed: output[" << i << "][" << j << "] = " << output[i * w + j] << std::endl;
                    pass = false;
                }
            }

            if (!pass) {
                break;
            }
        }

        if (pass) {
            std::cout << "Case: float passed!" << std::endl;
        }
        else {
            std::cout << "Case: float failed!" << std::endl;
        }

        cudaDestroyTextureObject(tex);
        cudaFree(output);
        d3dTexture->Release();
    }

    {
        // Init test data
        const int w = 16;
        const int h = 16;

        uchar4 input[h * w] = {
            {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
            {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2},
            {3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3},
            {4, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4}};
        uchar4 *output;
        cudaMallocManaged(&output, sizeof(input));

        // Create a texture for DirectX and CUDA interoperability
        ID3D11Texture2D *d3dTexture = create_d3d11_tex(d3dDevice, w, h, DXGI_FORMAT_R8G8B8A8_UNORM);

        // Init interop
        cudaGraphicsResource_t cudaResource = initInterop(d3dTexture);
        
        // Get the mapped array from the CUDA resource
        cudaArray_t cudaArr;
        cudaGraphicsSubResourceGetMappedArray(&cudaArr, cudaResource, 0, 0);

        // Access the underlying memory of interop CUDA resource
        cudaMemcpy2DToArray(cudaArr, 0, 0, input, sizeof(uchar4) * w,
                            sizeof(uchar4) * w, h, cudaMemcpyHostToDevice);
        cudaTextureObject_t tex = getTex(cudaArr);

        dim3 blockSize(4, 4);
        dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
        accessTexture<<<gridSize, blockSize>>>(output, tex, w, h);
        cudaDeviceSynchronize();

        cleanupInterop(cudaResource);

        bool pass = true;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (output[i * w + j].x != input[i * w + j].x ||
                    output[i * w + j].y != input[i * w + j].y ||
                    output[i * w + j].z != input[i * w + j].z ||
                    output[i * w + j].w != input[i * w + j].w) {
                    std::cout << "Failed: output[" << i << "][" << j << "] = {"
                              << output[i * w + j].x << ", "
                              << output[i * w + j].y << ", "
                              << output[i * w + j].z << ", "
                              << output[i * w + j].w << "}"
                              << std::endl;
                    pass = false;
                }
            }

            if (!pass) {
                break;
            }
        }

        if (pass) {
            std::cout << "Case: uchar4 passed!" << std::endl;
        }
        else {
            std::cout << "Case: uchar4 failed!" << std::endl;
        }

        cudaDestroyTextureObject(tex);
        cudaFree(output);
        d3dTexture->Release();
    }

    // DX cleanup
    d3dContext->Release();
    d3dDevice->Release();
    pFactory1->Release();
    pAdapter1->Release();

    cudaDeviceReset();

    return 0;
}
