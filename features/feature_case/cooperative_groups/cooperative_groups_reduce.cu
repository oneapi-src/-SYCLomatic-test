// ===------- cooperative_groups_reduce.cu --------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <numeric>
#include <algorithm>
namespace cg = cooperative_groups;

__global__ void CGReduce(int *input, int *out, unsigned int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    // cooperative_groups::reduce(tile, )
    auto tid = block.thread_rank();
    int thread_sum = input[tid];
    int thread_min = input[tid];
    int thread_max = input[tid];
    int thread_bit_and = input[tid];
    int thread_bit_xor = input[tid];
    int thread_bit_or = input[tid];
    for (int i = tid + tile32.size(); i < n; i+=tile32.size()) {
        thread_sum += input[i];
        thread_min = input[i] < thread_min ? input[i] : thread_min;
        thread_max = input[i] > thread_max ? input[i] : thread_max;
        thread_bit_and = input[i] & thread_bit_and;
        thread_bit_xor = input[i] ^ thread_bit_xor;
        thread_bit_or =  input[i] | thread_bit_or;

    }
    int plus = cooperative_groups::reduce(tile32, thread_sum, cg::plus<int>());
    int less = cooperative_groups::reduce(tile32, thread_min, cg::less<int>());
    int greater = cooperative_groups::reduce(tile32, thread_max, cg::greater<int>());
    int bit_and = cooperative_groups::reduce(tile32, thread_bit_and, cg::bit_and<int>());
    int bit_xor = cooperative_groups::reduce(tile32, thread_bit_xor, cg::bit_xor<int>());
    int bit_or = cooperative_groups::reduce(tile32, thread_bit_or, cg::bit_or<int>());
    out[0] = plus;
    out[1] = less;
    out[2] = greater;
    out[3] = bit_and;
    out[4] = bit_xor;
    out[5] = bit_or;
    if (tile32.meta_group_rank() == 0) {
        out[6] = 11;
        out[7] = tile32.shfl_up(input[3], 2);
        out[8] = tile32.shfl_down(input[3], 2);
        out[9] = tile32.shfl_xor(input[3], 2);
        out[10] = tile32.shfl(input[3], 0);
    }
}
template <class T>
void init_value(T * input, size_t size, T &max, T &min) {

    for (int i =0; i < size; i++) {
        input[i] = i;
        max = std::max(max, input[i]);
        min = std::min(min, input[i]);
    }
}
int main () {
    size_t array_size = 60;
    void *d_input;
    void *d_output;
    size_t input_size = array_size * sizeof(int);

    int *host_input = (int *)malloc(input_size);
    int *host_out = (int *)malloc(input_size);
    int ret = 0;
    int max = 0;
    int min = 0;
    init_value<int>(host_input, array_size, max, min);
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, input_size);
    cudaMemcpy(d_input, host_input, input_size, cudaMemcpyHostToDevice);
    void *kernelFunc[] = {(void *)d_input, (void *)d_output, (void *)&array_size};
    dim3 DimGrid = {128, 1, 1};
    dim3 DimBlock = {32, 1, 1};
    CGReduce<<<DimGrid, DimBlock>>>((int *)d_input, (int *)d_output, array_size);
    cudaMemcpy(host_out, d_output, input_size, cudaMemcpyDeviceToHost);
    int sum = std::reduce(host_input, host_input + array_size,0, std::plus<int>());
    int bit_and = std::reduce(host_input, host_input + array_size,0, std::bit_and<int>());
    int bit_xor = std::reduce(host_input, host_input + array_size,0, std::bit_xor<int>());
    int bit_or = std::reduce(host_input, host_input + array_size,0, std::bit_or<int>());

    bool is_result_expected = true;
    if (sum != host_out[0]) {
        std::cout << "Same result" << sum << " " << host_out[0] <<std::endl;
        is_result_expected = false;    
    }
    if (min != host_out[1]) {
        std::cout << "less result" << min << " " << host_out[1] <<std::endl;
        is_result_expected = false;    
    }
    if (max != host_out[2]) {
        std::cout << "greater result" << max << " " << host_out[2]<< std::endl;
        is_result_expected = false;    
    }
    if (bit_and != host_out[3]) {
        std::cout << "bit_and result" << bit_and << " " << host_out[3] <<std::endl;
        is_result_expected = false;    
    }
    if (bit_xor != host_out[4]) {
        std::cout << "bit_xor result" << bit_xor << " " << host_out[4] <<std::endl;
        is_result_expected = false;    
    }
    if (bit_or != host_out[5]) {
        std::cout << "bit_or result" << bit_or << " " << host_out[5] <<std::endl;
        is_result_expected = false;    
    }
    if (11 != host_out[6])
    {
        std::cout << "meta_group_rank execution failed" << std::endl;
        is_result_expected = false;
    }
    if (3 != host_out[7])
    {
        std::cout << "shfl_up execution failed " << std::endl;
        is_result_expected = false;
    }
    if (3 != host_out[8])
    {
        std::cout << "shfl_down execution failed " << std::endl;

        is_result_expected = false;
    }
    if (3 != host_out[9])
    {
        std::cout << "shfl_xor execution failed " << std::endl;

        is_result_expected = false;
    }
    if (3 != host_out[10])
    {
        std::cout << "shfl execution failed " << std::endl;

        is_result_expected = false;
    }
    std::cout <<"Other " << host_out[6] << " " << host_out[7] << " " << host_out[8] << " " << host_out[9]<< " " << host_out[10] << std::endl;
    if (!is_result_expected) {
        return -1;
    }
    std::cout << "pass" << std::endl;
    return 0;

}
