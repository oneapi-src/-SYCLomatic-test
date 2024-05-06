// ====------ cub_block.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>

#define WARP_SIZE 32
#define DATA_NUM 256

void init_data(int* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = i;
}
bool verify_data(int* data, int* expect, int num, int step = 1) {
  for(int i = 0; i < num; i = i + step) {
    if(data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}
void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}

__global__ void BlockExclusiveScanKernel(int* data) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).ExclusiveScan(input, output, 0, cub::Sum());
  data[threadid] = output;
}

__global__ void BlockExclusiveSumKernel(int* data) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).ExclusiveSum(input, output);
  data[threadid] = output;
}

__global__ void BlockInclusiveScanKernel(int* data) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).InclusiveScan(input, output, cub::Sum());
  data[threadid] = output;
}

__global__ void BlockInclusiveSumKernel(int* data) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).InclusiveSum(input, output);
  data[threadid] = output;
}

__global__ void BlockSumKernel(int* data) {
  typedef cub::BlockReduce<int, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 8, 1> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = BlockReduce(temp1).Sum(input);
  data[threadid] = output;
}

__global__ void BlockReduceKernel(int* data) {
  typedef cub::BlockReduce<int, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 8, 1> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = BlockReduce(temp1).Reduce(input, cub::Sum());
  data[threadid] = output;
}

template <int BLOCK_THREADS>
__global__ void ReduceSumValid(int *d_result, int *d_data, int valid_items,
                               int num_elements) {
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;

  int value = 0;
  if (threadIdx.x < BLOCK_THREADS) {
    if (blockIdx.x * BLOCK_THREADS + threadIdx.x < num_elements)
      value = d_data[blockIdx.x * BLOCK_THREADS + threadIdx.x];
  }

  int aggregate = BlockReduce(smem_storage).Sum(value, valid_items);
  if (threadIdx.x == 0)
    d_result[blockIdx.x] = aggregate;
  return;
}

bool TestBlockReduceSumValidItem() {
  srand(19850814);
  const int block_threads = 128;
  int num_elements = 1024;
  int valid_items = 64;
  int num_blocks = num_elements / block_threads +
                   (num_elements % block_threads == 0 ? 0 : 1);
  std::vector<int> data_vec(num_elements, 0);
  std::vector<int> result_vec(num_blocks, 0);
  std::vector<int> result_ref_vec(num_blocks, 0);
  std::vector<int> result_ref_cnt_vec(num_blocks, 0);
  int *data = data_vec.data();
  int *result = result_vec.data();
  int *result_ref = result_ref_vec.data();
  int *result_ref_cnt = result_ref_cnt_vec.data();

  int *d_data;
  int *d_result;
  cudaMalloc((void **)&d_data, sizeof(int) * num_elements);
  cudaMalloc((void **)&d_result, sizeof(int) * num_blocks);
  for (int i = 0; i < num_elements; ++i) {
    data[i] = rand() % num_elements;
    int block_id = i / block_threads;
    result_ref[block_id] +=
        result_ref_cnt[block_id] < valid_items ? data[i] : 0;
    result_ref_cnt_vec[block_id]++;
  }
  cudaMemcpy(d_data, data, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(block_threads, 1, 1);
  ReduceSumValid<block_threads>
      <<<dimGrid, dimBlock>>>(d_result, d_data, valid_items, num_elements);
  cudaMemcpy(result, d_result, num_blocks * sizeof(int),
             cudaMemcpyDeviceToHost);

  bool flag = true;
  for (int i = 0; i < num_blocks; ++i) {
    if (result[i] != result_ref[i]) {
      flag = false;
      printf("the %dth block result is wrong: %d %d\n", i + 1, result[i],
             result_ref[i]);
    }
  }

  cudaFree(d_data);
  cudaFree(d_result);
  return flag;
}

int main() {
  bool Result = true;
  int* dev_data = nullptr;
  dim3 GridSize;
  dim3 BlockSize;
  cudaMallocManaged(&dev_data, DATA_NUM * sizeof(int));
  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect1[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  init_data(dev_data, DATA_NUM);

  BlockExclusiveScanKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect1, DATA_NUM)) {
    std::cout << "BlockExclusiveScanKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect1, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect2[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  init_data(dev_data, DATA_NUM);

  BlockExclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect2, DATA_NUM)) {
    std::cout << "BlockExclusiveSumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect2, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect3[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  init_data(dev_data, DATA_NUM);

  BlockInclusiveScanKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect3, DATA_NUM)) {
    std::cout << "BlockInclusiveScanKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect3, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect4[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  init_data(dev_data, DATA_NUM);

  BlockInclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect4, DATA_NUM)) {
    std::cout << "BlockInclusiveSumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect4, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect5[DATA_NUM] = {
    8128, 496, 495, 493, 490, 486, 481, 475, 468, 460, 451, 441, 430, 418, 405, 391, 376, 360, 343, 325, 306, 286, 265, 243, 220, 196, 171, 145, 118, 90, 61, 31,
    1520, 1488, 1455, 1421, 1386, 1350, 1313, 1275, 1236, 1196, 1155, 1113, 1070, 1026, 981, 935, 888, 840, 791, 741, 690, 638, 585, 531, 476, 420, 363, 305, 246, 186, 125, 63,
    2544, 2480, 2415, 2349, 2282, 2214, 2145, 2075, 2004, 1932, 1859, 1785, 1710, 1634, 1557, 1479, 1400, 1320, 1239, 1157, 1074, 990, 905, 819, 732, 644, 555, 465, 374, 282, 189, 95,
    3568, 3472, 3375, 3277, 3178, 3078, 2977, 2875, 2772, 2668, 2563, 2457, 2350, 2242, 2133, 2023, 1912, 1800, 1687, 1573, 1458, 1342, 1225, 1107, 988, 868, 747, 625, 502, 378, 253, 127,
    24512, 4464, 4335, 4205, 4074, 3942, 3809, 3675, 3540, 3404, 3267, 3129, 2990, 2850, 2709, 2567, 2424, 2280, 2135, 1989, 1842, 1694, 1545, 1395, 1244, 1092, 939, 785, 630, 474, 317, 159,
    5616, 5456, 5295, 5133, 4970, 4806, 4641, 4475, 4308, 4140, 3971, 3801, 3630, 3458, 3285, 3111, 2936, 2760, 2583, 2405, 2226, 2046, 1865, 1683, 1500, 1316, 1131, 945, 758, 570, 381, 191,
    6640, 6448, 6255, 6061, 5866, 5670, 5473, 5275, 5076, 4876, 4675, 4473, 4270, 4066, 3861, 3655, 3448, 3240, 3031, 2821, 2610, 2398, 2185, 1971, 1756, 1540, 1323, 1105, 886, 666, 445, 223,
    7664, 7440, 7215, 6989, 6762, 6534, 6305, 6075, 5844, 5612, 5379, 5145, 4910, 4674, 4437, 4199, 3960, 3720, 3479, 3237, 2994, 2750, 2505, 2259, 2012, 1764, 1515, 1265, 1014, 762, 509, 255
  };
  init_data(dev_data, DATA_NUM);

  BlockSumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect5, DATA_NUM, 128)) {
    std::cout << "BlockSumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect5, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect6[DATA_NUM] = {
    8128, 496, 495, 493, 490, 486, 481, 475, 468, 460, 451, 441, 430, 418, 405, 391, 376, 360, 343, 325, 306, 286, 265, 243, 220, 196, 171, 145, 118, 90, 61, 31,
    1520, 1488, 1455, 1421, 1386, 1350, 1313, 1275, 1236, 1196, 1155, 1113, 1070, 1026, 981, 935, 888, 840, 791, 741, 690, 638, 585, 531, 476, 420, 363, 305, 246, 186, 125, 63,
    2544, 2480, 2415, 2349, 2282, 2214, 2145, 2075, 2004, 1932, 1859, 1785, 1710, 1634, 1557, 1479, 1400, 1320, 1239, 1157, 1074, 990, 905, 819, 732, 644, 555, 465, 374, 282, 189, 95,
    3568, 3472, 3375, 3277, 3178, 3078, 2977, 2875, 2772, 2668, 2563, 2457, 2350, 2242, 2133, 2023, 1912, 1800, 1687, 1573, 1458, 1342, 1225, 1107, 988, 868, 747, 625, 502, 378, 253, 127,
    24512, 4464, 4335, 4205, 4074, 3942, 3809, 3675, 3540, 3404, 3267, 3129, 2990, 2850, 2709, 2567, 2424, 2280, 2135, 1989, 1842, 1694, 1545, 1395, 1244, 1092, 939, 785, 630, 474, 317, 159,
    5616, 5456, 5295, 5133, 4970, 4806, 4641, 4475, 4308, 4140, 3971, 3801, 3630, 3458, 3285, 3111, 2936, 2760, 2583, 2405, 2226, 2046, 1865, 1683, 1500, 1316, 1131, 945, 758, 570, 381, 191,
    6640, 6448, 6255, 6061, 5866, 5670, 5473, 5275, 5076, 4876, 4675, 4473, 4270, 4066, 3861, 3655, 3448, 3240, 3031, 2821, 2610, 2398, 2185, 1971, 1756, 1540, 1323, 1105, 886, 666, 445, 223,
    7664, 7440, 7215, 6989, 6762, 6534, 6305, 6075, 5844, 5612, 5379, 5145, 4910, 4674, 4437, 4199, 3960, 3720, 3479, 3237, 2994, 2750, 2505, 2259, 2012, 1764, 1515, 1265, 1014, 762, 509, 255
  };
  init_data(dev_data, DATA_NUM);

  BlockReduceKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect6, DATA_NUM, 128)) {
    std::cout << "BlockReduceKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect6, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  Result  = TestBlockReduceSumValidItem() && Result;

  if(Result) {
    std::cout << "passed" << std::endl;
    return 0;
  }
  return 1;
}


