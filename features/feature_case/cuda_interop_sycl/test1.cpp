

#include <sycl/sycl.hpp>
#include <cuda_runtime.h>


__global__ void vecAdd(double *a, double *b, double *c, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    c[id] = a[id] + b[id];
  }
}

int main(int argc, char *argv[]) {
  using namespace sycl;
  int n = 100;
  size_t bytes = n * sizeof(double);

  device dev{};
  context myContext{dev};
  queue myQueue{myContext, dev};

  // Allocate memory for each vector on host
  auto d_A = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  auto d_B = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  auto d_C = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
for (int i = 0; i < n; i++) {
    std::cout <<d_C[i]<<'\t';
  }
  // Initialize vectors on host
  for (int i = 0; i < n; i++) {
    d_A[i] = std::sin(i) * std::sin(i);
    d_B[i] = std::cos(i) * std::cos(i);
  }
  cudaStream_t stream_cuda;
  cudaStreamCreate(&stream_cuda);

  myQueue.submit([&](handler& h) {
        int blockSize = 1024;
        int gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));
        vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
  });

  int blockSize = 1024;
  int gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));
  vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
  cudaDeviceSynchronize();

  myQueue.wait();

  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += d_C[i];
  }
  std::cout << "Final result " << sum / n << std::endl;

  free(d_A, myContext);
  free(d_B, myContext);
  free(d_C, myContext);

  return 0;
}