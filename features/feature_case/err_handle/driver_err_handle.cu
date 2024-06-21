#include <cuda.h>
#include <iostream>

int main() {
  CUresult e;
  const char *err_s;
  cuGetErrorString(e, &err_s);
  std::cout << err_s << std::endl;
  return 0;
}
