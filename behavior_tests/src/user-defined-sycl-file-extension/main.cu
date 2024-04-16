#include "src.h"
#include "cuda_src.cuh"

int main(void) {
    do_nothing_kernel<<<1,1>>>();
    return 0;
}
