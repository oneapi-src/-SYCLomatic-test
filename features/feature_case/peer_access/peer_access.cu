#include <cuda_runtime.h>
#include <iostream>

int main() {
  int cur_device;
  cudaGetDevice(&cur_device);
  
  int can_access = 0;
  cudaDeviceCanAccessPeer(&can_access, cur_device, cur_device);
  cudaDeviceDisablePeerAccess(cur_device);
  cudaDeviceEnablePeerAccess(cur_device, 0);

  std::cout << "test passed" << std::endl;

  return 0;
}