// Modifications Copyright (C) 2023 Intel Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <random>
#include "cudaSift.h"
#include "cudautils.h"

//================= Device matching functions =====================//

void MatchSiftPoints(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2,
                     const sycl::nd_item<3> &item_ct1, float *siftPoint,
                     float *sums)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int p1 = item_ct1.get_group(2);
  const int p2 = item_ct1.get_group(1) * 16 + ty;
  const float *ptr1 = sift1[p1].data;
  const float *ptr2 = sift2[p2].data;
  const int i = 16 * ty + tx;
  if (ty < 8)
    siftPoint[i] = ptr1[i];
  /*
  DPCT1065:146: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float sum = 0.0f;
  if (p2 < numPts2)
    for (int j = 0; j < 8; j++)
      sum += siftPoint[16 * j + tx] * ptr2[16 * j + tx];
  sums[i] = sum;
  /*
  DPCT1065:147: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx < 8)
    sums[i] += sums[i + 8];
  /*
  DPCT1065:148: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx < 4)
    sums[i] += sums[i + 4];
  /*
  DPCT1065:149: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (ty == 0)
  {
    sum = sums[16 * tx + 0] + sums[16 * tx + 1] + sums[16 * tx + 2] + sums[16 * tx + 3];
    corrData[p1 * item_ct1.get_group_range(1) * 16 +
             item_ct1.get_group(1) * 16 + tx] = sum;
  }
  /*
  DPCT1065:150: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
}

void MatchSiftPoints2(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2,
                      const sycl::nd_item<3> &item_ct1, float *siftPoints1,
                      float *siftPoints2)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const float *ptr1 =
      sift1[dpct::min(numPts1 - 1,
                      (unsigned int)(item_ct1.get_group(2) * 16 + ty))]
          .data;
  const float *ptr2 =
      sift2[dpct::min(numPts2 - 1,
                      (unsigned int)(item_ct1.get_group(1) * 16 + ty))]
          .data;
  for (int i = 0; i < 8; i++)
  {
    siftPoints1[128 * ty + 16 * i + tx] = ptr1[16 * i + tx];
    siftPoints2[128 * ty + 16 * i + tx] = ptr2[16 * i + tx];
  }
  /*
  DPCT1065:151: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  const int p1 = item_ct1.get_group(2) * 16 + ty;
  const int p2 = item_ct1.get_group(1) * 16 + tx;
  const float *pt1 = &siftPoints1[ty * 128];
  const float *pt2 = &siftPoints2[tx * 128];
  float sum = 0.0f;
  for (int i = 0; i < 128; i++)
  {
    int itx = (i + tx) & 127; // avoid bank conflicts
    sum += pt1[itx] * pt2[itx];
  }
  if (p1 < numPts1)
    corrData[p1 * item_ct1.get_group_range(1) * 16 + p2] =
        (p2 < numPts2 ? sum : -1.0f);
}

void FindMaxCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize,
                 const sycl::nd_item<3> &item_ct1, float *maxScore,
                 float *maxScor2, int *maxIndex)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int idx = ty * 16 + tx;
  int p1 = item_ct1.get_group(2) * 16 + item_ct1.get_local_id(1);
  p1 = (p1 >= numPts1 ? numPts1 - 1 : p1);
  maxScore[idx] = -1.0f;
  maxScor2[idx] = -1.0f;
  maxIndex[idx] = -1;
  /*
  DPCT1065:152: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float *corrs = &corrData[p1 * corrWidth];
  for (int i = tx; i < corrWidth; i += 16)
  {
    float val = corrs[i];
    if (val > maxScore[idx])
    {
      maxScor2[idx] = maxScore[idx];
      maxScore[idx] = val;
      maxIndex[idx] = i;
    }
    else if (val > maxScor2[idx])
      maxScor2[idx] = val;
  }
  /*
  DPCT1065:153: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  for (int len = 8; len > 0; len /= 2)
  {
    if (tx < 8)
    {
      float val = maxScore[idx + len];
      int i = maxIndex[idx + len];
      if (val > maxScore[idx])
      {
        maxScor2[idx] = maxScore[idx];
        maxScore[idx] = val;
        maxIndex[idx] = i;
      }
      else if (val > maxScor2[idx])
        maxScor2[idx] = val;
      float va2 = maxScor2[idx + len];
      if (va2 > maxScor2[idx])
        maxScor2[idx] = va2;
    }
    /*
    DPCT1118:49: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:154: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  if (tx == 0)
  {
    sift1[p1].score = maxScore[ty * 16];
    sift1[p1].ambiguity = maxScor2[ty * 16] / (maxScore[ty * 16] + 1e-6);
    sift1[p1].match = maxIndex[ty * 16];
    sift1[p1].match_xpos = sift2[maxIndex[ty * 16]].xpos;
    sift1[p1].match_ypos = sift2[maxIndex[ty * 16]].ypos;
  }
}

// Version based on suggestion by Nicholas Lin
void FindMaxCorr3(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, int *maxIndex)
{
  int block_dim = item_ct1.get_local_range(2); // blockDim.x == 16
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int p1 = item_ct1.get_group(2) * block_dim + ty;
  const int idx = ty * 16 + tx;

  maxIndex[idx] = 0;
  /*
  DPCT1065:155: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  float *corrs = NULL;
  if (p1 < numPts1)
  {
    corrs = &corrData[p1 * block_dim * 2];
    corrs[tx] = 0.0f;
    corrs[tx + 16] = 0.0f;
    const float *pt1 = sift1[p1].data;
    for (int p2 = tx; p2 < numPts2; p2 += 16)
    {
      float *pt2 = sift2[p2].data;
      float sum = 0.0f;
      for (int i = 0; i < 128; i++)
        sum += pt1[i] * pt2[i];
      if (sum > corrs[tx])
      {
        corrs[tx + 16] = corrs[tx];
        corrs[tx] = sum;
        maxIndex[idx] = p2;
      }
      else if (sum > corrs[tx + 16])
        corrs[tx + 16] = sum;
    }
  }
  /*
  DPCT1065:156: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (p1 < numPts1)
  {
    for (int len = 8; len > 0; len /= 2)
    {
      if (tx < len)
      {
        float val = corrs[tx + len];
        int i = maxIndex[idx + len];
        if (val > corrs[tx])
        {
          corrs[tx + 16] = corrs[tx];
          corrs[tx] = val;
          maxIndex[idx] = i;
        }
        else if (val > corrs[tx + 16])
          corrs[tx + 16] = val;
        float va2 = corrs[tx + 16 + len];
        if (va2 > corrs[tx + 16])
          corrs[tx + 16] = va2;
      }
      /*
      DPCT1118:50: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:157: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    if (tx == 0)
    {
      sift1[p1].score = corrs[0];
      sift1[p1].ambiguity = corrs[16] / (corrs[0] + 1e-6);
      sift1[p1].match = maxIndex[ty << 4];
      sift1[p1].match_xpos = sift2[maxIndex[ty << 4]].xpos;
      sift1[p1].match_ypos = sift2[maxIndex[ty << 4]].ypos;
    }
  }
}

#define FMC2W 16
#define FMC2H 4

void FindMaxCorr2(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, float *siftPoint,
                  float *maxScore, float *maxScor2, int *maxIndex)
{

  const int p1 = item_ct1.get_group(2);
  if (p1 >= numPts1)
    return;
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int idx = ty * FMC2W + tx;
  if (idx < FMC2H)
  {
    maxScore[idx] = -1.0f;
    maxScor2[idx] = -1.0f;
    maxIndex[idx] = 0;
  }
  /*
  DPCT1065:158: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  const float *pt1 = sift1[p1].data;
  for (int i = idx; i < 128; i += FMC2W * FMC2H)
    siftPoint[i] = pt1[i];
  /*
  DPCT1065:159: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  for (int p2 = ty; p2 < numPts2; p2 += FMC2H)
  {
    const float *pt2 = sift2[p2].data;
    float sum = 0.0f;
    for (int j = tx; j < 128; j += FMC2W)
      sum += siftPoint[j] * pt2[j];
    for (int j = FMC2W / 2; j > 0; j /= 2)
      sum += ShiftDown(sum, j, item_ct1);
    if (tx == 0)
    {
      if (sum > maxScore[ty])
      {
        maxScor2[ty] = maxScore[ty];
        maxScore[ty] = sum;
        maxIndex[ty] = p2;
      }
      else if (sum > maxScor2[ty])
        maxScor2[ty] = sum;
    }
  }
  /*
  DPCT1065:160: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  for (int len = FMC2H / 2; len > 0; len /= 2)
  {
    if (ty == 0 && tx < len)
    {
      float val = maxScore[tx + len];
      int p2 = maxIndex[tx + len];
      if (val > maxScore[tx])
      {
        maxScor2[tx] = maxScore[tx];
        maxScore[tx] = val;
        maxIndex[tx] = p2;
      }
      else if (val > maxScor2[tx])
        maxScor2[tx] = val;
      float va2 = maxScor2[tx + len];
      if (va2 > maxScor2[tx])
        maxScor2[tx] = va2;
    }
    /*
    DPCT1118:51: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:161: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  if (ty == 0 && tx == 0)
  {
    sift1[p1].score = maxScore[0];
    sift1[p1].ambiguity = maxScor2[0] / (maxScore[0] + 1e-6);
    sift1[p1].match = maxIndex[0];
    sift1[p1].match_xpos = sift2[maxIndex[0]].xpos;
    sift1[p1].match_ypos = sift2[maxIndex[0]].ypos;
  }
}

void FindMaxCorr4(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, float *siftPoint,
                  float *maxScore, float *maxScor2, int *maxIndex)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  if (tx == 0)
  {
    maxScore[ty] = -1.0f;
    maxScor2[ty] = -1.0f;
    maxIndex[ty] = 0;
  }
  const int p1 = item_ct1.get_group(2) * FMC2H + ty;
  const float *pt1 = sift1[p1].data;
  for (int j = tx; j < 128; j += FMC2W)
    siftPoint[128 * ty + j] = pt1[j];
  /*
  DPCT1065:162: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  for (int p2 = 0; p2 < numPts2; p2++)
  {
    const float *pt2 = sift2[p2].data;
    float sum = 0.0f;
    for (int j = tx; j < 128; j += FMC2W)
      sum += siftPoint[128 * ty + j] * pt2[j];
    for (int j = FMC2W / 2; j > 0; j /= 2)
      sum += ShiftDown(sum, j, item_ct1);
    if (tx == 0)
    {
      if (sum > maxScore[ty])
      {
        maxScor2[ty] = maxScore[ty];
        maxScore[ty] = sum;
        maxIndex[ty] = p2;
      }
      else if (sum > maxScor2[ty])
        maxScor2[ty] = sum;
    }
  }
  /*
  DPCT1065:163: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx == 0)
  {
    sift1[p1].score = maxScore[ty];
    sift1[p1].ambiguity = maxScor2[ty] / (maxScore[ty] + 1e-6);
    sift1[p1].match = maxIndex[ty];
    sift1[p1].match_xpos = sift2[maxIndex[ty]].xpos;
    sift1[p1].match_ypos = sift2[maxIndex[ty]].ypos;
  }
}

void memcopyKernel(float *src, float *dst, size_t src_pitch, size_t dst_pitch, int numPts, size_t width)
{
  char *d_src = (char *)src;
  char *d_dst = (char *)dst;

  for (int i = 0; i < numPts; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      d_dst[j] = d_src[j];
    }
    d_src = d_src + src_pitch;
    d_dst = d_dst + dst_pitch;
  }
}

void
CleanMatches(SiftPoint *sift1, int numPts1, const sycl::nd_item<3> &item_ct1)
{
  const int p1 = dpct::min(
      (unsigned int)(item_ct1.get_group(2) * 64 + item_ct1.get_local_id(2)),
      numPts1 - 1);
  sift1[p1].score = 0.0f;
}

#define M7W 32
#define M7H 32
#define M7R 4
#define NRX 2
#define NDIM 128

/*
DPCT1110:52: The total declared local variable size in device function
FindMaxCorr10 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                   const sycl::nd_item<3> &item_ct1, sycl::float4 *buffer1,
                   sycl::float4 *buffer2)
{

  int tx = item_ct1.get_local_id(2);
  int ty = item_ct1.get_local_id(1);
  int bp1 = M7W * item_ct1.get_group(2);
  for (int j = ty; j < M7W; j += M7H / M7R)
  {
    int p1 = sycl::min(bp1 + j, numPts1 - 1);
    for (int d = tx; d < NDIM / 4; d += M7W)
      buffer1[j * NDIM / 4 + (d + j) % (NDIM / 4)] =
          ((sycl::float4 *)&sift1[p1].data)[d];
  }

  float max_score[NRX];
  float sec_score[NRX];
  int index[NRX];
  for (int i = 0; i < NRX; i++)
  {
    max_score[i] = 0.0f;
    sec_score[i] = 0.0f;
    index[i] = -1;
  }

  int idx = ty * M7W + tx;
  int ix = idx % (M7W / NRX);
  int iy = idx / (M7W / NRX);
  for (int bp2 = 0; bp2 < numPts2 - M7H + 1; bp2 += M7H)
  {
    for (int j = ty; j < M7H; j += M7H / M7R)
    {
      int p2 = sycl::min(bp2 + j, numPts2 - 1);
      for (int d = tx; d < NDIM / 4; d += M7W)
        buffer2[j * NDIM / 4 + d] = ((sycl::float4 *)&sift2[p2].data)[d];
    }
    /*
    DPCT1118:53: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:165: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (idx < M7W * M7H / M7R / NRX)
    {
      float score[M7R][NRX];
      for (int dy = 0; dy < M7R; dy++)
        for (int i = 0; i < NRX; i++)
          score[dy][i] = 0.0f;
      for (int d = 0; d < NDIM / 4; d++)
      {
        sycl::float4 v1[NRX];
        for (int i = 0; i < NRX; i++)
          v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM / 4 + (d + (M7W / NRX) * i + ix) % (NDIM / 4)];
        for (int dy = 0; dy < M7R; dy++)
        {
          sycl::float4 v2 = buffer2[(M7R * iy + dy) * (NDIM / 4) + d];
          for (int i = 0; i < NRX; i++)
          {
            score[dy][i] += v1[i].x() * v2.x();
            score[dy][i] += v1[i].y() * v2.y();
            score[dy][i] += v1[i].z() * v2.z();
            score[dy][i] += v1[i].w() * v2.w();
          }
        }
      }
      for (int dy = 0; dy < M7R; dy++)
      {
        for (int i = 0; i < NRX; i++)
        {
          if (score[dy][i] > max_score[i])
          {
            sec_score[i] = max_score[i];
            max_score[i] = score[dy][i];
            index[i] = sycl::min(bp2 + M7R * iy + dy, numPts2 - 1);
          }
          else if (score[dy][i] > sec_score[i])
            sec_score[i] = score[dy][i];
        }
      }
    }
    /*
    DPCT1118:54: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:166: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  float *scores1 = (float *)buffer1;
  float *scores2 = &scores1[M7W * M7H / M7R];
  int *indices = (int *)&scores2[M7W * M7H / M7R];
  if (idx < M7W * M7H / M7R / NRX)
  {
    for (int i = 0; i < NRX; i++)
    {
      scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
      scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
      indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
    }
  }
  /*
  DPCT1065:164: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (ty == 0)
  {
    float max_score = scores1[tx];
    float sec_score = scores2[tx];
    int index = indices[tx];
    for (int y = 0; y < M7H / M7R; y++)
      if (index != indices[y * M7W + tx])
      {
        if (scores1[y * M7W + tx] > max_score)
        {
          sec_score = sycl::max(max_score, sec_score);
          max_score = scores1[y * M7W + tx];
          index = indices[y * M7W + tx];
        }
        else if (scores1[y * M7W + tx] > sec_score)
          sec_score = scores1[y * M7W + tx];
      }
    sift1[bp1 + tx].score = max_score;
    sift1[bp1 + tx].match = index;
    sift1[bp1 + tx].match_xpos = sift2[index].xpos;
    sift1[bp1 + tx].match_ypos = sift2[index].ypos;
    sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
  }
}

#define FMC_GH 512
#define FMC_BW 32
#define FMC_BH 32
#define FMC_BD 16
#define FMC_TW 1
#define FMC_TH 4
#define FMC_NW (FMC_BW / FMC_TW) //  32
#define FMC_NH (FMC_BH / FMC_TH) //   8
#define FMC_NT (FMC_NW * FMC_NH) // 256 = 8 warps

dpct::global_memory<volatile int, 0> lock(0);

/*
DPCT1110:55: The total declared local variable size in device function
FindMaxCorr9 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void FindMaxCorr9(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, volatile int &lock,
                  sycl::float4 *siftParts1, sycl::float4 *siftParts2)
{
   // 4*32*8 = 1024
   // 4*32*8 = 1024
  //__shared__ float blksums[FMC_BW*FMC_BH];     // 32*32  = 1024
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int idx = ty * FMC_NW + tx;
  sycl::float4 *pts1 = 0, *pts2 = 0;
  if (idx < FMC_BW)
  {
    const int p1l = dpct::min(
        (unsigned int)(item_ct1.get_group(2) * FMC_BW + idx), numPts1 - 1);
    pts1 = (sycl::float4 *)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < sycl::min(FMC_GH, numPts2 - FMC_BH + 1); k += FMC_BH)
  {
    if (idx < FMC_BH)
    {
      const int p2l =
          dpct::min((unsigned int)(item_ct1.get_group(1) * FMC_GH + k + idx),
                    numPts2 - 1);
      pts2 = (sycl::float4 *)sift2[p2l].data;
    }
    float sums[FMC_TW * FMC_TH];
    for (int i = 0; i < FMC_TW * FMC_TH; i++)
      sums[i] = 0.0f;

    if (idx < FMC_BW)
      for (int i = 0; i < FMC_BD / 2; i++)
        siftParts1[(i + 0) * FMC_BW + idx] = pts1[0 + i];
    if (idx < FMC_BH)
      for (int i = 0; i < FMC_BD / 2; i++)
        siftParts2[(i + 0) * FMC_BH + idx] = pts2[0 + i];
    /*
    DPCT1118:56: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:169: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int b = FMC_BD / 2;
    for (int d = FMC_BD / 2; d < 32; d += FMC_BD / 2)
    {
      if (idx < FMC_BW)
        for (int i = 0; i < FMC_BD / 2; i++)
          siftParts1[(i + b) * FMC_BW + idx] = pts1[d + i];
      if (idx < FMC_BH)
        for (int i = 0; i < FMC_BD / 2; i++)
          siftParts2[(i + b) * FMC_BH + idx] = pts2[d + i];

      b ^= FMC_BD / 2;
      for (int i = 0; i < FMC_BD / 2; i++)
      {
        sycl::float4 v1[FMC_TW];
        for (int ix = 0; ix < FMC_TW; ix++)
          v1[ix] = siftParts1[(i + b) * FMC_BW + (tx * FMC_TW + ix)];
        for (int iy = 0; iy < FMC_TH; iy++)
        {
          sycl::float4 v2 = siftParts2[(i + b) * FMC_BH + (ty * FMC_TH + iy)];
          for (int ix = 0; ix < FMC_TW; ix++)
          {
            sums[iy * FMC_TW + ix] += v1[ix].x() * v2.x();
            sums[iy * FMC_TW + ix] += v1[ix].y() * v2.y();
            sums[iy * FMC_TW + ix] += v1[ix].z() * v2.z();
            sums[iy * FMC_TW + ix] += v1[ix].w() * v2.w();
          }
        }
      }
      /*
      DPCT1118:60: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:173: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }

    b ^= FMC_BD / 2;
    for (int i = 0; i < FMC_BD / 2; i++)
    {
      sycl::float4 v1[FMC_TW];
      for (int ix = 0; ix < FMC_TW; ix++)
        v1[ix] = siftParts1[(i + b) * FMC_BW + (tx * FMC_TW + ix)];
      for (int iy = 0; iy < FMC_TH; iy++)
      {
        sycl::float4 v2 = siftParts2[(i + b) * FMC_BH + (ty * FMC_TH + iy)];
        for (int ix = 0; ix < FMC_TW; ix++)
        {
          sums[iy * FMC_TW + ix] += v1[ix].x() * v2.x();
          sums[iy * FMC_TW + ix] += v1[ix].y() * v2.y();
          sums[iy * FMC_TW + ix] += v1[ix].z() * v2.z();
          sums[iy * FMC_TW + ix] += v1[ix].w() * v2.w();
        }
      }
    }
    /*
    DPCT1118:57: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:170: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    float *blksums = (float *)siftParts1;
    for (int iy = 0; iy < FMC_TH; iy++)
      for (int ix = 0; ix < FMC_TW; ix++)
        blksums[(ty * FMC_TH + iy) * FMC_BW + (tx * FMC_TW + ix)] = sums[iy * FMC_TW + ix];
    /*
    DPCT1118:58: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:171: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (idx < FMC_BW)
    {
      for (int j = 0; j < FMC_BH; j++)
      {
        float sum = blksums[j * FMC_BW + idx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex =
              dpct::min((unsigned int)(item_ct1.get_group(1) * FMC_GH + k + j),
                        numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    /*
    DPCT1118:59: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:172: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  const int p1 = dpct::min((unsigned int)(item_ct1.get_group(2) * FMC_BW + idx),
                           numPts1 - 1);
  if (idx == 0)
    while (dpct::atomic_compare_exchange_strong<
               sycl::access::address_space::generic_space>((int *)&lock, 0,
                                                           1) != 0)
      ;
  /*
  DPCT1065:167: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < FMC_BW)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = sycl::max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  /*
  DPCT1065:168: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx == 0)
    dpct::atomic_exchange<sycl::access::address_space::generic_space>(
        (int *)&lock, 0);
}

/*
DPCT1110:61: The total declared local variable size in device function
FindMaxCorr8 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void FindMaxCorr8(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, volatile int &lock,
                  sycl::float4 *siftParts1, sycl::float4 *siftParts2,
                  float *blksums)
{
   // 4*32*8 = 1024
   // 4*32*8 = 1024
   // 32*32  = 1024
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int idx = ty * FMC_NW + tx;
  sycl::float4 *pts1 = 0, *pts2 = 0;
  if (idx < FMC_BW)
  {
    const int p1l = dpct::min(
        (unsigned int)(item_ct1.get_group(2) * FMC_BW + idx), numPts1 - 1);
    pts1 = (sycl::float4 *)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < sycl::min(FMC_GH, numPts2 - FMC_BH + 1); k += FMC_BH)
  {
    if (idx < FMC_BH)
    {
      const int p2l =
          dpct::min((unsigned int)(item_ct1.get_group(1) * FMC_GH + k + idx),
                    numPts2 - 1);
      pts2 = (sycl::float4 *)sift2[p2l].data;
    }
    float sums[FMC_TW * FMC_TH];
    for (int i = 0; i < FMC_TW * FMC_TH; i++)
      sums[i] = 0.0f;
    for (int d = 0; d < 32; d += FMC_BD)
    {
      if (idx < FMC_BW)
        for (int i = 0; i < FMC_BD; i++)
          siftParts1[i * FMC_BW + idx] = pts1[d + i];
      if (idx < FMC_BH)
        for (int i = 0; i < FMC_BD; i++)
          siftParts2[i * FMC_BH + idx] = pts2[d + i];
      /*
      DPCT1118:64: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:178: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();

      for (int i = 0; i < FMC_BD; i++)
      {
        sycl::float4 v1[FMC_TW];
        for (int ix = 0; ix < FMC_TW; ix++)
          v1[ix] = siftParts1[i * FMC_BW + (tx * FMC_TW + ix)];
        for (int iy = 0; iy < FMC_TH; iy++)
        {
          sycl::float4 v2 = siftParts2[i * FMC_BH + (ty * FMC_TH + iy)];
          for (int ix = 0; ix < FMC_TW; ix++)
          {
            sums[iy * FMC_TW + ix] += v1[ix].x() * v2.x();
            sums[iy * FMC_TW + ix] += v1[ix].y() * v2.y();
            sums[iy * FMC_TW + ix] += v1[ix].z() * v2.z();
            sums[iy * FMC_TW + ix] += v1[ix].w() * v2.w();
          }
        }
      }
      /*
      DPCT1118:65: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:179: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    // float *blksums = (float*)siftParts1;
    for (int iy = 0; iy < FMC_TH; iy++)
      for (int ix = 0; ix < FMC_TW; ix++)
        blksums[(ty * FMC_TH + iy) * FMC_BW + (tx * FMC_TW + ix)] = sums[iy * FMC_TW + ix];
    /*
    DPCT1118:62: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:176: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (idx < FMC_BW)
    {
      for (int j = 0; j < FMC_BH; j++)
      {
        float sum = blksums[j * FMC_BW + idx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex =
              dpct::min((unsigned int)(item_ct1.get_group(1) * FMC_GH + k + j),
                        numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    /*
    DPCT1118:63: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:177: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  const int p1 = dpct::min((unsigned int)(item_ct1.get_group(2) * FMC_BW + idx),
                           numPts1 - 1);
  if (idx == 0)
    while (dpct::atomic_compare_exchange_strong<
               sycl::access::address_space::generic_space>((int *)&lock, 0,
                                                           1) != 0)
      ;
  /*
  DPCT1065:174: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < FMC_BW)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = sycl::max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  /*
  DPCT1065:175: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx == 0)
    dpct::atomic_exchange<sycl::access::address_space::generic_space>(
        (int *)&lock, 0);
}

/*
DPCT1110:66: The total declared local variable size in device function
FindMaxCorr7 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void FindMaxCorr7(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, volatile int &lock,
                  float *siftParts1, float *siftParts2)
{
   // features in columns
   // one extra to avoid shared conflicts
  sycl::float4 *pts1 = (sycl::float4 *)siftParts1;
  sycl::float4 *pts2 = (sycl::float4 *)siftParts2;
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int p1l =
      dpct::min((unsigned int)(item_ct1.get_group(2) * 16 + ty), numPts1 - 1);
  const sycl::float4 *p1l4 = (sycl::float4 *)sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < 512 / 16; k++)
  {
    const int p2l = dpct::min(
        (unsigned int)(item_ct1.get_group(1) * 512 + k * 16 + ty), numPts2 - 1);
    const sycl::float4 *p2l4 = (sycl::float4 *)sift2[p2l].data;
#define NUM 4
    float sum[NUM];
    if (ty < (16 / NUM))
      for (int l = 0; l < NUM; l++)
        sum[l] = 0.0f;
    /*
    DPCT1118:67: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:182: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = 0; i < 2; i++)
    {
      pts1[17 * tx + ty] = p1l4[i * 16 + tx];
      pts2[16 * ty + tx] = p2l4[i * 16 + tx];
      /*
      DPCT1118:70: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:185: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      if (ty < (16 / NUM))
      {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
          sycl::float4 p1v = pts1[17 * j + tx];
#pragma unroll
          for (int l = 0; l < NUM; l++)
          {
            sycl::float4 p2v = pts2[16 * (ty + l * (16 / NUM)) + j];
            sum[l] += p1v.x() * p2v.x();
            sum[l] += p1v.y() * p2v.y();
            sum[l] += p1v.z() * p2v.z();
            sum[l] += p1v.w() * p2v.w();
          }
        }
      }
      /*
      DPCT1118:71: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:186: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    float *sums = siftParts1;
    if (ty < (16 / NUM))
      for (int l = 0; l < NUM; l++)
        sums[16 * (ty + l * (16 / NUM)) + tx] = sum[l];
    /*
    DPCT1118:68: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:183: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (ty == 0)
    {
      for (int j = 0; j < 16; j++)
      {
        float sum = sums[16 * j + tx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = dpct::min(
              (unsigned int)(item_ct1.get_group(1) * 512 + k * 16 + j),
              numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    /*
    DPCT1118:69: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:184: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  const int p1 =
      dpct::min((unsigned int)(item_ct1.get_group(2) * 16 + tx), numPts1 - 1);
  if (tx == 0 && ty == 0)
    while (dpct::atomic_compare_exchange_strong<
               sycl::access::address_space::generic_space>((int *)&lock, 0,
                                                           1) != 0)
      ;
  /*
  DPCT1065:180: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (ty == 0)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = sycl::max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  /*
  DPCT1065:181: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx == 0 && ty == 0)
    dpct::atomic_exchange<sycl::access::address_space::generic_space>(
        (int *)&lock, 0);
}

void FindMaxCorr6(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, volatile int &lock,
                  float *siftParts2, float *sums)
{
  //__shared__ float siftParts1[128*16]; // features in columns
  // one extra to avoid shared conflicts

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int p1l =
      dpct::min((unsigned int)(item_ct1.get_group(2) * 16 + ty), numPts1 - 1);
  float *pt1l = sift1[p1l].data;
  sycl::float4 part1 = reinterpret_cast<sycl::float4 *>(pt1l)[tx];
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < 512; k += 16)
  {
    const int p2l = dpct::min(
        (unsigned int)(item_ct1.get_group(1) * 512 + k + ty), numPts2 - 1);
    float *pt2l = sift2[p2l].data;
    reinterpret_cast<sycl::float4 *>(siftParts2)[32 * ty + tx] =
        reinterpret_cast<sycl::float4 *>(pt2l)[tx];
    /*
    DPCT1118:72: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:189: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = 0; i < 16; i++)
    {
      sycl::float4 part2 =
          reinterpret_cast<sycl::float4 *>(siftParts2)[32 * i + tx];
      float sum = part1.x() * part2.x() + part1.y() * part2.y() +
                  part1.z() * part2.z() + part1.w() * part2.w();
      sum += ShiftDown(sum, 16, item_ct1);
      sum += ShiftDown(sum, 8, item_ct1);
      sum += ShiftDown(sum, 4, item_ct1);
      sum += ShiftDown(sum, 2, item_ct1);
      sum += ShiftDown(sum, 1, item_ct1);
      if (tx == 0)
        sums[16 * i + ty] = sum;
    }
    /*
    DPCT1118:73: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:190: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (ty == 0 && tx < 16)
    {
      for (int j = 0; j < 16; j++)
      {
        float sum = sums[16 * j + tx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = dpct::min(
              (unsigned int)(item_ct1.get_group(1) * 512 + k + j), numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    /*
    DPCT1118:74: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:191: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  if (tx == 0 && ty == 0)
    while (dpct::atomic_compare_exchange_strong<
               sycl::access::address_space::generic_space>((int *)&lock, 0,
                                                           1) != 0)
      ;
  /*
  DPCT1065:187: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (ty == 0 && tx < 16)
  {
    const int p1 =
        dpct::min((unsigned int)(item_ct1.get_group(2) * 16 + tx), numPts1 - 1);
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = sycl::max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  /*
  DPCT1065:188: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx == 0 && ty == 0)
    dpct::atomic_exchange<sycl::access::address_space::generic_space>(
        (int *)&lock, 0);
}

void FindMaxCorr5(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2,
                  const sycl::nd_item<3> &item_ct1, volatile int &lock,
                  float *siftParts1, float *siftParts2)
{
   // features in columns
   // one extra to avoid shared conflicts
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int p1l =
      dpct::min((unsigned int)(item_ct1.get_group(2) * 16 + ty), numPts1 - 1);
  const float *pt1l = sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < 512 / 16; k++)
  {
    const int p2l = dpct::min(
        (unsigned int)(item_ct1.get_group(1) * 512 + k * 16 + ty), numPts2 - 1);
    const float *pt2l = sift2[p2l].data;
    float sum = 0.0f;
    for (int i = 0; i < 8; i++)
    {
      siftParts1[17 * tx + ty] = pt1l[i * 16 + tx]; // load and transpose
      siftParts2[17 * tx + ty] = pt2l[i * 16 + tx];
      /*
      DPCT1118:77: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:196: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      for (int j = 0; j < 16; j++)
        sum += siftParts1[17 * j + tx] * siftParts2[17 * j + ty];
      /*
      DPCT1118:78: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:197: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    float *sums = siftParts1;
    sums[16 * ty + tx] = sum;
    /*
    DPCT1118:75: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (ty == 0)
    {
      for (int j = 0; j < 16; j++)
      {
        float sum = sums[16 * j + tx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = dpct::min(
              (unsigned int)(item_ct1.get_group(1) * 512 + k * 16 + j),
              numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    /*
    DPCT1118:76: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:195: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  const int p1 =
      dpct::min((unsigned int)(item_ct1.get_group(2) * 16 + tx), numPts1 - 1);
  if (tx == 0 && ty == 0)
    while (dpct::atomic_compare_exchange_strong<
               sycl::access::address_space::generic_space>((int *)&lock, 0,
                                                           1) != 0)
      ;
  /*
  DPCT1065:192: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (ty == 0)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = sycl::max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  /*
  DPCT1065:193: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx == 0 && ty == 0)
    dpct::atomic_exchange<sycl::access::address_space::generic_space>(
        (int *)&lock, 0);
}

template <int size>
/*
DPCT1110:79: The total declared local variable size in device function
InvertMatrix exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void InvertMatrix(float elem[size][size], float res[size][size])
{
  int indx[size];
  float b[size];
  float vv[size];
  for (int i = 0; i < size; i++)
    indx[i] = 0;
  int imax = 0;
  float d = 1.0;
  for (int i = 0; i < size; i++)
  { // find biggest element for each row
    float big = 0.0;
    for (int j = 0; j < size; j++)
    {
      float temp = sycl::fabs(elem[i][j]);
      if (temp > big)
        big = temp;
    }
    if (big > 0.0)
      vv[i] = 1.0 / big;
    else
      vv[i] = 1e16;
  }
  for (int j = 0; j < size; j++)
  {
    for (int i = 0; i < j; i++)
    {                                   // i<j
      float sum = elem[i][j];           // i<j (lower left)
      for (int k = 0; k < i; k++)       // k<i<j
        sum -= elem[i][k] * elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum;                 // i<j (lower left)
    }
    float big = 0.0;
    for (int i = j; i < size; i++)
    {                                   // i>=j
      float sum = elem[i][j];           // i>=j (upper right)
      for (int k = 0; k < j; k++)       // k<j<=i
        sum -= elem[i][k] * elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum;                 // i>=j (upper right)
      float dum = vv[i] * sycl::fabs(sum);
      if (dum >= big)
      {
        big = dum;
        imax = i;
      }
    }
    if (j != imax)
    { // imax>j
      for (int k = 0; k < size; k++)
      {
        float dum = elem[imax][k]; // upper right and lower left
        elem[imax][k] = elem[j][k];
        elem[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (elem[j][j] == 0.0) // j==j (upper right)
      elem[j][j] = 1e-16;
    if (j != (size - 1))
    {
      float dum = 1.0 / elem[j][j];
      for (int i = j + 1; i < size; i++) // i>j
        elem[i][j] *= dum;               // i>j (upper right)
    }
  }
  for (int j = 0; j < size; j++)
  {
    for (int k = 0; k < size; k++)
      b[k] = 0.0;
    b[j] = 1.0;
    int ii = -1;
    for (int i = 0; i < size; i++)
    {
      int ip = indx[i];
      float sum = b[ip];
      b[ip] = b[i];
      if (ii != -1)
        for (int j = ii; j < i; j++)
          sum -= elem[i][j] * b[j]; // i>j (upper right)
      else if (sum != 0.0)
        ii = i;
      b[i] = sum;
    }
    for (int i = size - 1; i >= 0; i--)
    {
      float sum = b[i];
      for (int j = i + 1; j < size; j++)
        sum -= elem[i][j] * b[j]; // i<j (lower left)
      b[i] = sum / elem[i][i];    // i==i (upper right)
    }
    for (int i = 0; i < size; i++)
      res[i][j] = b[i];
  }
}

/*
DPCT1110:80: The total declared local variable size in device function
ComputeHomographies exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void ComputeHomographies(float *coord, int *randPts, float *homo, int numPts,
                         const sycl::nd_item<3> &item_ct1)
{
  float a[8][8], ia[8][8];
  float b[8];
  const int bx = item_ct1.get_group(2);
  const int tx = item_ct1.get_local_id(2);
  const int idx = item_ct1.get_local_range(2) * bx + tx;
  const int numLoops =
      item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
  for (int i = 0; i < 4; i++)
  {
    int pt = randPts[i * numLoops + idx];
    float x1 = coord[pt + 0 * numPts];
    float y1 = coord[pt + 1 * numPts];
    float x2 = coord[pt + 2 * numPts];
    float y2 = coord[pt + 3 * numPts];
    float *row1 = a[2 * i + 0];
    row1[0] = x1;
    row1[1] = y1;
    row1[2] = 1.0;
    row1[3] = row1[4] = row1[5] = 0.0;
    row1[6] = -x2 * x1;
    row1[7] = -x2 * y1;
    float *row2 = a[2 * i + 1];
    row2[0] = row2[1] = row2[2] = 0.0;
    row2[3] = x1;
    row2[4] = y1;
    row2[5] = 1.0;
    row2[6] = -y2 * x1;
    row2[7] = -y2 * y1;
    b[2 * i + 0] = x2;
    b[2 * i + 1] = y2;
  }
  InvertMatrix<8>(a, ia);
  item_ct1.barrier(sycl::access::fence_space::local_space);
  for (int j = 0; j < 8; j++)
  {
    float sum = 0.0f;
    for (int i = 0; i < 8; i++)
      sum += ia[j][i] * b[i];
    homo[j * numLoops + idx] = sum;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

#define TESTHOMO_TESTS 16 // number of tests per block,  alt. 32, 32
#define TESTHOMO_LOOPS 16 // number of loops per block,  alt.  8, 16

void TestHomographies(float *d_coord, float *d_homo,
                                 int *d_counts, int numPts, float thresh2,
                                 const sycl::nd_item<3> &item_ct1, float *homo,
                                 int *cnts)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int idx = item_ct1.get_group(1) * item_ct1.get_local_range(1) + tx;
  const int numLoops =
      item_ct1.get_local_range(1) * item_ct1.get_group_range(1);
  if (ty < 8 && tx < TESTHOMO_LOOPS)
    homo[tx * 8 + ty] = d_homo[idx + ty * numLoops];
  item_ct1.barrier(sycl::access::fence_space::local_space);
  float a[8];
  for (int i = 0; i < 8; i++)
    a[i] = homo[ty * 8 + i];
  int cnt = 0;
  for (int i = tx; i < numPts; i += TESTHOMO_TESTS)
  {
    float x1 = d_coord[i + 0 * numPts];
    float y1 = d_coord[i + 1 * numPts];
    float x2 = d_coord[i + 2 * numPts];
    float y2 = d_coord[i + 3 * numPts];
    /*
    DPCT1013:198: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    float nomx = a[0] * x1 + a[1] * y1 + a[2];
    /*
    DPCT1013:199: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    float nomy = a[3] * x1 + a[4] * y1 + a[5];
    /*
    DPCT1013:200: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    float deno = a[6] * x1 + a[7] * y1 + 1.0f;
    /*
    DPCT1013:201: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    float errx = x2 * deno - nomx;
    /*
    DPCT1013:202: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    float erry = y2 * deno - nomy;
    /*
    DPCT1013:203: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    float err2 = errx * errx + erry * erry;
    /*
    DPCT1013:204: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    if (err2 < thresh2 * deno * deno)
      cnt++;
  }
  int kty = TESTHOMO_TESTS * ty;
  cnts[kty + tx] = cnt;
  item_ct1.barrier(sycl::access::fence_space::local_space);
  int len = TESTHOMO_TESTS / 2;
  while (len > 0)
  {
    if (tx < len)
      cnts[kty + tx] += cnts[kty + tx + len];
    len /= 2;
    /*
    DPCT1118:81: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
  if (tx < TESTHOMO_LOOPS && ty == 0)
    d_counts[idx] = cnts[TESTHOMO_TESTS * tx];
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

//================= Host matching functions =====================//

double FindHomography(SiftData &data, float *homography, int *numMatches, float &matchTime, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  *numMatches = 0;
  homography[0] = homography[4] = homography[8] = 1.0f;
  homography[1] = homography[2] = homography[3] = 0.0f;
  homography[5] = homography[6] = homography[7] = 0.0f;
  if (data.d_data == NULL)
    return 0.0f;
  SiftPoint *d_sift = data.d_data;
  numLoops = iDivUp(numLoops, 16) * 16;
  int numPts = data.numPts;
  if (numPts < 8)
    return 0.0f;
  int numPtsUp = iDivUp(numPts, 16) * 16;
  float *d_coord, *d_homo;
  int *d_randPts, *h_randPts;
  int randSize = 4 * sizeof(int) * numLoops;
  int szFl = sizeof(float);
  int szPt = sizeof(SiftPoint);
#ifdef DEVICE_TIMER
  auto start_malloc = std::chrono::steady_clock::now();
#endif
  safeCall(DPCT_CHECK_ERROR(
      d_coord = (float *)sycl::malloc_device(4 * sizeof(float) * numPtsUp,
                                             dpct::get_in_order_queue())));
  safeCall(DPCT_CHECK_ERROR(d_randPts = (int *)sycl::malloc_device(
                                randSize, dpct::get_in_order_queue())));
  safeCall(DPCT_CHECK_ERROR(
      d_homo = (float *)sycl::malloc_device(8 * sizeof(float) * numLoops,
                                            dpct::get_in_order_queue())));

#ifdef DEVICE_TIMER
  auto stop_malloc = std::chrono::steady_clock::now();
  matchTime += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif
  h_randPts = (int *)malloc(randSize);
  float *h_scores = (float *)malloc(sizeof(float) * numPtsUp);
  float *h_ambiguities = (float *)malloc(sizeof(float) * numPtsUp);

  // temp variables are for host memory allocation, device data is transferred to temp
  float *temp1 = (float *)malloc(szPt * numPtsUp);
  float *temp2 = (float *)malloc(szPt * numPtsUp);

#ifdef DEVICE_TIMER
  auto start_memcpy_1 = std::chrono::steady_clock::now();
#endif

  safeCall(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(temp1, &d_sift[0].score, szPt * numPts)
                                .wait()));
  safeCall(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(temp2, &d_sift[0].ambiguity, szPt * numPts)
                           .wait()));

#ifdef DEVICE_TIMER
  auto stop_memcpy_1 = std::chrono::steady_clock::now();
  matchTime += std::chrono::duration<float, std::micro>(stop_memcpy_1 - start_memcpy_1).count();
#endif

  char *src_score = (char *)temp1;
  char *src_ambiguity = (char *)temp2;
  char *dst_score = (char *)h_scores;
  char *dst_ambiguity = (char *)h_ambiguities;

  for (int i = 0; i < numPts; ++i)
  {
    memcpy(dst_score, src_score, szFl);
    memcpy(dst_ambiguity, src_ambiguity, szFl);
    src_score += szPt;
    src_ambiguity += szPt;
    dst_score += szFl;
    dst_ambiguity += szFl;
  }

  int *validPts = (int *)malloc(sizeof(int) * numPts);
  int numValid = 0;
  for (int i = 0; i < numPts; i++)
  {
    if (h_scores[i] > minScore && h_ambiguities[i] < maxAmbiguity)
      validPts[numValid++] = i;
  }
  free(h_scores);
  free(h_ambiguities);
  if (numValid >= 8)
  {
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
    for (int i = 0; i < numLoops; i++)
    {
      int p1 = dis(rnd) % numValid;
      int p2 = dis(rnd) % numValid;
      int p3 = dis(rnd) % numValid;
      int p4 = dis(rnd) % numValid;
      while (p2 == p1)
        p2 = dis(rnd) % numValid;
      while (p3 == p1 || p3 == p2)
        p3 = dis(rnd) % numValid;
      while (p4 == p1 || p4 == p2 || p4 == p3)
        p4 = dis(rnd) % numValid;
      h_randPts[i + 0 * numLoops] = validPts[p1];
      h_randPts[i + 1 * numLoops] = validPts[p2];
      h_randPts[i + 2 * numLoops] = validPts[p3];
      h_randPts[i + 3 * numLoops] = validPts[p4];
    }

    float *temp3, *temp4, *temp5, *temp6;
#ifdef DEVICE_TIMER
    auto start_malloc_2 = std::chrono::steady_clock::now();
#endif
    safeCall(
        DPCT_CHECK_ERROR(temp3 = (float *)sycl::malloc_device(
                             szPt * numPtsUp, dpct::get_in_order_queue())));
    safeCall(
        DPCT_CHECK_ERROR(temp4 = (float *)sycl::malloc_device(
                             szPt * numPtsUp, dpct::get_in_order_queue())));
    safeCall(
        DPCT_CHECK_ERROR(temp5 = (float *)sycl::malloc_device(
                             szPt * numPtsUp, dpct::get_in_order_queue())));
    safeCall(
        DPCT_CHECK_ERROR(temp6 = (float *)sycl::malloc_device(
                             szPt * numPtsUp, dpct::get_in_order_queue())));
#ifdef DEVICE_TIMER
    auto stop_malloc_2 = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_malloc_2 - start_malloc_2).count();
#endif
#ifdef DEVICE_TIMER
    auto start_memcpy_2 = std::chrono::steady_clock::now();
#endif
    safeCall(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(d_randPts, h_randPts, randSize)
                                  .wait()));
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    safeCall(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
        temp3, &d_sift[0].xpos, szPt * numPts)));
    safeCall(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
        temp4, &d_sift[0].ypos, szPt * numPts)));
    safeCall(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
        temp5, &d_sift[0].match_xpos, szPt * numPts)));
    safeCall(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
        temp6, &d_sift[0].match_ypos, szPt * numPts)));

    // kernel calto transfer memory from device to device
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      float *d_coord_numPtsUp_ct1 = &d_coord[0 * numPtsUp];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            memcopyKernel(temp3, d_coord_numPtsUp_ct1, szPt, szFl, numPts,
                          szFl);
          });
    });
    /*
    DPCT1010:205: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    safeCall(0);
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      float *d_coord_numPtsUp_ct1 = &d_coord[1 * numPtsUp];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            memcopyKernel(temp4, d_coord_numPtsUp_ct1, szPt, szFl, numPts,
                          szFl);
          });
    });
    /*
    DPCT1010:206: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    safeCall(0);
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      float *d_coord_numPtsUp_ct1 = &d_coord[2 * numPtsUp];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            memcopyKernel(temp5, d_coord_numPtsUp_ct1, szPt, szFl, numPts,
                          szFl);
          });
    });
    /*
    DPCT1010:207: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    safeCall(0);
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      float *d_coord_numPtsUp_ct1 = &d_coord[3 * numPtsUp];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            memcopyKernel(temp6, d_coord_numPtsUp_ct1, szPt, szFl, numPts,
                          szFl);
          });
    });
    /*
    DPCT1010:208: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    safeCall(0);
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

#ifdef DEVICE_TIMER
    auto stop_memcpy_2 = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_memcpy_2 - start_memcpy_2).count();
#endif
#ifdef DEVICE_TIMER
    auto start_kernel_1 = std::chrono::steady_clock::now();
#endif
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, numLoops / 16) *
                              sycl::range<3>(1, 1, 16),
                          sycl::range<3>(1, 1, 16)),
        [=](sycl::nd_item<3> item_ct1) {
          ComputeHomographies(d_coord, d_randPts, d_homo, numPtsUp, item_ct1);
        });
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
    auto stop_kernel_1 = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_kernel_1 - start_kernel_1).count();
    // printf("ComputeHomographies time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel_1 - start_kernel_1).count());
#endif
    checkMsg("ComputeHomographies() execution failed\n");

    sycl::range<3> blocks(1, numLoops / TESTHOMO_LOOPS, 1);
    sycl::range<3> threads(1, TESTHOMO_LOOPS, TESTHOMO_TESTS);
#ifdef DEVICE_TIMER
    auto start_kernel_2 = std::chrono::steady_clock::now();
#endif
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:210: '8 * TESTHOMO_LOOPS' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<float, 1> homo_acc_ct1(
          sycl::range<1>(128 /*8 * TESTHOMO_LOOPS*/), cgh);
      /*
      DPCT1101:211: 'TESTHOMO_TESTS * TESTHOMO_LOOPS' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<int, 1> cnts_acc_ct1(
          sycl::range<1>(256 /*TESTHOMO_TESTS * TESTHOMO_LOOPS*/), cgh);

      cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                         TestHomographies(d_coord, d_homo, d_randPts, numPtsUp,
                                          thresh * thresh, item_ct1,
                                          homo_acc_ct1.get_pointer(),
                                          cnts_acc_ct1.get_pointer());
                       });
    });
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
    auto stop_kernel_2 = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_kernel_2 - start_kernel_2).count();
    // printf("TestHomographies time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel_2 - start_kernel_2).count());
#endif
    checkMsg("TestHomographies() execution failed\n");
#ifdef DEVICE_TIMER
    auto start_memcpy_3 = std::chrono::steady_clock::now();
#endif
    safeCall(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(h_randPts, d_randPts, sizeof(int) * numLoops)
            .wait()));
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
    auto stop_memcpy_3 = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_memcpy_3 - start_memcpy_3).count();
#endif
    int maxIndex = -1, maxCount = -1;
    for (int i = 0; i < numLoops; i++)
      if (h_randPts[i] > maxCount)
      {
        maxCount = h_randPts[i];
        maxIndex = i;
      }

    *numMatches = maxCount;
#ifdef DEVICE_TIMER
    auto start_memcpy_4 = std::chrono::steady_clock::now();
#endif
    safeCall(DPCT_CHECK_ERROR(dpct::dpct_memcpy(
        homography, szFl, &d_homo[maxIndex], sizeof(float) * numLoops, szFl, 8,
        dpct::device_to_host)));
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
    auto stop_memcpy_4 = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_memcpy_4 - start_memcpy_4).count();
#endif

    safeCall(DPCT_CHECK_ERROR(sycl::free(temp3, dpct::get_in_order_queue())));
    safeCall(DPCT_CHECK_ERROR(sycl::free(temp4, dpct::get_in_order_queue())));
    safeCall(DPCT_CHECK_ERROR(sycl::free(temp5, dpct::get_in_order_queue())));
    safeCall(DPCT_CHECK_ERROR(sycl::free(temp6, dpct::get_in_order_queue())));
  }

  free(validPts);
  free(h_randPts);
  free(temp1);
  free(temp2);

  safeCall(DPCT_CHECK_ERROR(sycl::free(d_homo, dpct::get_in_order_queue())));
  safeCall(DPCT_CHECK_ERROR(sycl::free(d_randPts, dpct::get_in_order_queue())));
  safeCall(DPCT_CHECK_ERROR(sycl::free(d_coord, dpct::get_in_order_queue())));
  return matchTime;
}

double MatchSiftData(SiftData &data1, SiftData &data2, float &matchTime)
{
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2)
    return 0.0;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data == NULL || data2.d_data == NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif

// Original version with correlation and maximization in two different kernels
// Global memory reguirement: O(N^2)
#if 0
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
#if 0 
  dim3 blocks1(numPts1, iDivUp(numPts2, 16));
  dim3 threads1(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks1, threads1>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#else 
  dim3 blocks(iDivUp(numPts1,16), iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 16 points x 16 points
  MatchSiftPoints2<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#endif
  safeCall(cudaDeviceSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Version suggested by Nicholas Lin with combined correlation and maximization
// Global memory reguirement: O(N)
#if 0
  int block_dim = 16;
  float *d_corrData;
  int corrSize = numPts1 * block_dim * 2;
  safeCall(cudaMalloc((void **)&d_corrData, sizeof(float) * corrSize));
  dim3 blocks(iDivUp(numPts1, block_dim));
  dim3 threads(block_dim, block_dim); 
  FindMaxCorr3<<<blocks, threads >>>(d_corrData, sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr3() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

#if 0
  dim3 blocksMax(numPts1);
  dim3 threadsMax(FMC2W, FMC2H);
  FindMaxCorr2<<<blocksMax, threadsMax>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr2() execution failed\n");
#endif

// Combined version with no global memory requirement using one FMC2H points per block
#if 0
  dim3 blocksMax2(iDivUp(numPts1, FMC2H));
  dim3 threadsMax2(FMC2W, FMC2H);
  FindMaxCorr4<<<blocksMax2, threadsMax2>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr4() execution failed\n");
#endif

// Combined version with no global memory requirement using global locks
#if 1
  sycl::range<3> blocksMax3(1, iDivUp(numPts2, 512), iDivUp(numPts1, 16));
  sycl::range<3> threadsMax3(1, 16, 16);
#ifdef DEVICE_TIMER
  auto start_kernel1 = std::chrono::steady_clock::now();
#endif
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, iDivUp(numPts1, 64)) *
                            sycl::range<3>(1, 1, 64),
                        sycl::range<3>(1, 1, 64)),
      [=](sycl::nd_item<3> item_ct1) {
        CleanMatches(sift1, numPts1, item_ct1);
      });
  /*
  DPCT1010:209: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  safeCall(0);
  safeCall(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
  auto stop_kernel1 = std::chrono::steady_clock::now();
  // printf("CleanMatches time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel1 - start_kernel1).count());
  matchTime += std::chrono::duration<float, std::micro>(stop_kernel1 - start_kernel1).count();
  auto matchSiftDataTime += std::chrono::duration<float, std::micro>(stop_kernel1 - start_kernel1).count();
#endif
  int mode = 10;
  // if (mode == 5) // K40c 5.0ms, 1080 Ti 1.2ms, 2080 Ti 0.83ms
  //   FindMaxCorr5<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  // else if (mode == 6)
  // { // 2080 Ti 0.89ms
  //   threadsMax3 = dim3(32, 16);
  //   FindMaxCorr6<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  // }
  // else if (mode == 7) // 2080 Ti 0.50ms
  //   FindMaxCorr7<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  // else if (mode == 8)
  // { // 2080 Ti 0.45ms
  //   blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
  //   threadsMax3 = dim3(FMC_NW, FMC_NH);
  //   FindMaxCorr8<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  // }
  // else if (mode == 9)
  // { // 2080 Ti 0.46ms
  //   blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
  //   threadsMax3 = dim3(FMC_NW, FMC_NH);
  //   FindMaxCorr9<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  // }
  // else
  if (mode == 10)
  {
    blocksMax3 = sycl::range<3>(1, 1, iDivUp(numPts1, M7W));
    threadsMax3 = sycl::range<3>(1, M7H / M7R, M7W);
#ifdef DEVICE_TIMER
    auto start_kernel2 = std::chrono::steady_clock::now();
#endif
    /*
    DPCT1049:82: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:212: 'M7W * NDIM / 4' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<sycl::float4, 1> buffer1_acc_ct1(
          sycl::range<1>(1024 /*M7W * NDIM / 4*/), cgh);
      /*
      DPCT1101:213: 'M7H * NDIM / 4' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<sycl::float4, 1> buffer2_acc_ct1(
          sycl::range<1>(1024 /*M7H * NDIM / 4*/), cgh);

      cgh.parallel_for(sycl::nd_range<3>(blocksMax3 * threadsMax3, threadsMax3),
                       [=](sycl::nd_item<3> item_ct1) {
                         FindMaxCorr10(sift1, sift2, numPts1, numPts2, item_ct1,
                                       buffer1_acc_ct1.get_pointer(),
                                       buffer2_acc_ct1.get_pointer());
                       });
    });
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
    auto stop_kernel2 = std::chrono::steady_clock::now();
    // printf("FindMaxCorr10 time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel2 - start_kernel2).count());
    matchTime += std::chrono::duration<float, std::micro>(stop_kernel2 - start_kernel2).count();
    matchSiftDataTime += std::chrono::duration<float, std::micro>(stop_kernel2 - start_kernel2).count();
#endif
  }
  checkMsg("FindMaxCorr10() execution failed\n");
#endif

  if (data1.h_data != NULL)
  {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
#ifdef DEVICE_TIMER
    auto start_memcpy = std::chrono::steady_clock::now();
#endif
    safeCall(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(h_ptr, d_ptr, sizeof(SiftPoint) * data1.numPts)
            .wait()));
    safeCall(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
#ifdef DEVICE_TIMER
    auto stop_memcpy = std::chrono::steady_clock::now();
    matchTime += std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
    matchSiftDataTime += std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
#endif
  }
  return matchTime;
}
