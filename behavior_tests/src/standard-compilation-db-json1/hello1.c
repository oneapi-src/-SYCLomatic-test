#include "cuda_runtime.h"

__global__ void k(){}

int main() {
  float2 f2;

  k<<<1,1>>>();

  return 0;
}
