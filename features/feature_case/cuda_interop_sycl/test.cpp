

#include <cuda_runtime.h>
#include <sycl/sycl.hpp>

// __global__ void vecAdd(double *a, double *b, double *c, int n) {
//   int id = blockIdx.x * blockDim.x + threadIdx.x;
//   if (id < n) {
//     c[id] = a[id] + b[id];
//   }
// }

void vecAdd(double *a, double *b, double *c, int n,
            const sycl::nd_item<3> &item_ct1) {
  // Get our global thread ID
  int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
           item_ct1.get_local_id(2);

  // Make sure we do not go out of bounds
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
  // auto d_A = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  // auto d_B = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  // auto d_C = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  double *d_A, *d_B, *d_C;
  double *h_A, *h_B, *h_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  h_A = new double(n);
  h_B = new double(n);
  h_C = new double(n);
  // Initialize vectors on host
  for (int i = 0; i < n; i++) {
    h_A[i] = 0.5;
    h_B[i] = 0.5;
  }
  cudaStream_t stream_cuda;
  cudaStreamCreate(&stream_cuda);
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  // myQueue.submit([&](handler& h) {
  //       int blockSize = 1024;
  //       int gridSize = static_cast<int>(ceil(static_cast<float>(n) /
  //       blockSize)); vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
  //       cudaDeviceSynchronize();
  // });

  // int blockSize = 1024;
  // int gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));
  // vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
  // cudaDeviceSynchronize();

  int blockSize = 1024;

  // Number of thread blocks in grid
  int gridSize = static_cast<int>((static_cast<float>(n) / blockSize));

  // Execute the kernel
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    myQueue.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *
                              sycl::range<3>(1, 1, blockSize),
                          sycl::range<3>(1, 1, blockSize)),
        [=](sycl::nd_item<3> item_ct1) { vecAdd(d_A, d_B, d_C, n, item_ct1); });
  }

  myQueue.wait();
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += h_C[i];
  }
  std::cout << "Final result " << sum / n << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
