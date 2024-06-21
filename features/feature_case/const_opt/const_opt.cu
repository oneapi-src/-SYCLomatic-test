#include<cuda_runtime.h>
#include<iostream>

__constant__ int const_a = 11;
__constant__ int const_b[32] = {1, 2, 3};
static __constant__ int const_c = 1;
static __constant__ int const_d[33] = {1, 2, 3};

__global__ void kernel1(int *ptr){
  *ptr = const_a + const_b[1] + const_c + const_d[2];
}


int main(){
  int *dev_a;
  cudaMalloc(&dev_a, sizeof(int));
  kernel1<<<1, 1>>>(dev_a);
  cudaDeviceSynchronize();
  int host_a;
  cudaMemcpy(&host_a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
  if(host_a != 17) {
    std::cout << "test failed" << std::endl;
    exit(-1);
  }
  std::cout << "test success" << std::endl;
  return 0;
}
