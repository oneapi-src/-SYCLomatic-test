#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to add a float value with itself
__global__ void addKernel(float *d_in, float *d_out) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    d_out[idx] = d_in[idx];
}

int main() {
    // Initialize float value
    float a = 3.0;
    float f_in = a;
    float f_out;

    // Allocate memory on the device
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(float));

    // Copy the value from host to device
    cudaMemcpy(d_in, &f_in, sizeof(float), cudaMemcpyHostToDevice);

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
    cudaMemcpy(&f_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Check the result
    printf("The sum of 3.0 and 3.0 is: %f\n", f_out);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

