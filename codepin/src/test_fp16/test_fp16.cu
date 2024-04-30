#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUDA kernel to square a half-precision floating-point number
__global__ void squareKernel(__half *d_in, __half *d_out) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    d_out[idx] = __hmul(d_in[idx], d_in[idx]);
}

int main() {
    // Initialize half-precision floating-point value
    __half h_in = __float2half(3.0f);
    __half h_out;

    // Allocate memory on the device
    __half *d_in, *d_out;
    cudaMalloc((void **)&d_in, sizeof(__half));
    cudaMalloc((void **)&d_out, sizeof(__half));

    // Copy the value from host to device
    cudaMemcpy(d_in, &h_in, sizeof(__half), cudaMemcpyHostToDevice);

    // Launch the kernel to square the value
    squareKernel<<<1, 1>>>(d_in, d_out);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "squareKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching squareKernel!\n", cudaStatus);
    }

    // Copy the result back to the host
    cudaMemcpy(&h_out, d_out, sizeof(__half), cudaMemcpyDeviceToHost);

    // Check the result
    float result = __half2float(h_out);
    printf("The square of 3.0 is: %f\n", result);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

