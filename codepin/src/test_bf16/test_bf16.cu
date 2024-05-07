#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUDA kernel to add a bfloat16 value with itself
__global__ void addKernel(nv_bfloat16 *d_in, nv_bfloat16 *d_out) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    d_out[idx] = d_in[idx];
}

int main() {
    // Initialize bfloat16 value
        float a= 3.0;
    nv_bfloat16 h_in = __float2bfloat16(a);
    nv_bfloat16 h_out;

    // Allocate memory on the device
    nv_bfloat16 *d_in, *d_out;
    cudaMalloc((void **)&d_in, sizeof(nv_bfloat16));
    cudaMalloc((void **)&d_out, sizeof(nv_bfloat16));

    // Copy the value from host to device
    cudaMemcpy(d_in, &h_in, sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

    // Launch the kernel to add the value with itself
    addKernel<<<1, 1>>>(d_in, d_out);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy the result back to the host
    cudaMemcpy(&h_out, d_out, sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

    // Check the result
    float result = __bfloat162float(h_out);
    printf("The sum of 3.0 and 3.0 is: %f\n", result);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

