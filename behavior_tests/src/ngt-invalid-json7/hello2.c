#include "cuda_runtime.h"

__global__ void k2(){}

int main() {
  float4 f4;

  k2<<<1,1>>>();

  return 0;
}
