#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test_ldg_tex_cache_read(int *deviceArray) {
  float f1;
  double d;
  float2 *f2;
  __half h1;
  __half2 *h2;
  uchar4 u4;
  ulonglong2 *ull2;

  __ldg(&f1);
  auto cacheReadD = __ldg(&d);
  __ldg(f2);
  auto cacheReadH1 = __ldg(&h1);
  __ldg(h2);
  __ldg(&u4);
  __ldg(ull2);
}

int main() {
  int test = 0;
  test_ldg_tex_cache_read<<<4, 4>>>(&test);
  cudaDeviceSynchronize();
  return 0;
}
