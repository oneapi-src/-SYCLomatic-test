//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

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
#include "cudautils.h"
#include "cudaSiftD.h"
#include "cudaSift.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

static dpct::constant_memory<int, 0> d_MaxNumPoints;
dpct::global_memory<unsigned int, 1> d_PointCounter(8 * 2 + 1);
static dpct::constant_memory<float, 1> d_ScaleDownKernel(5);
static dpct::constant_memory<float, 1> d_LowPassKernel(2 * LOWPASS_R + 1);
static dpct::constant_memory<float, 1> d_LaplaceKernel(8 * 12 * 16);

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter and subsample image
///////////////////////////////////////////////////////////////////////////////
void ScaleDownDenseShift(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch,
                         const sycl::nd_item<3> &item_ct1,
                         float const *d_ScaleDownKernel, float *brows)
{
#define BW (SCALEDOWN_W + 4)
#define BH (SCALEDOWN_H + 4)
#define W2 (SCALEDOWN_W / 2)
#define H2 (SCALEDOWN_H / 2)

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int xp = item_ct1.get_group(2) * SCALEDOWN_W + tx;
  const int yp = item_ct1.get_group(1) * SCALEDOWN_H + ty;
  const float k0 = d_ScaleDownKernel[0];
  const float k1 = d_ScaleDownKernel[1];
  const float k2 = d_ScaleDownKernel[2];
  const int xl = sycl::min(width - 1, sycl::max(0, xp - 2));
  const int yl = sycl::min(height - 1, sycl::max(0, yp - 2));
  if (xp < (width + 4) && yp < (height + 4))
  {
    float v = d_Data[yl * pitch + xl];
    brows[BW * ty + tx] =
        k0 * (v + ShiftDown(v, 4, item_ct1)) +
        k1 * (ShiftDown(v, 1, item_ct1) + ShiftDown(v, 3, item_ct1)) +
        k2 * ShiftDown(v, 2, item_ct1);
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  const int xs = item_ct1.get_group(2) * W2 + tx;
  const int ys = item_ct1.get_group(1) * H2 + ty;
  if (tx < W2 && ty < H2 && xs < (width / 2) && ys < (height / 2))
  {
    float *ptr = &brows[BW * (ty * 2) + (tx * 2)];
    d_Result[ys * newpitch + xs] = k0 * (ptr[0] + ptr[4 * BW]) + k1 * (ptr[1 * BW] + ptr[3 * BW]) + k2 * ptr[2 * BW];
  }
}

void ScaleDownDense(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch,
                    const sycl::nd_item<3> &item_ct1,
                    float const *d_ScaleDownKernel, float *irows, float *brows)
{
#define BW (SCALEDOWN_W + 4)
#define BH (SCALEDOWN_H + 4)
#define W2 (SCALEDOWN_W / 2)
#define H2 (SCALEDOWN_H / 2)

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int xp = item_ct1.get_group(2) * SCALEDOWN_W + tx;
  const int yp = item_ct1.get_group(1) * SCALEDOWN_H + ty;
  const int xl = sycl::min(width - 1, sycl::max(0, xp - 2));
  const int yl = sycl::min(height - 1, sycl::max(0, yp - 2));
  const float k0 = d_ScaleDownKernel[0];
  const float k1 = d_ScaleDownKernel[1];
  const float k2 = d_ScaleDownKernel[2];
  if (xp < (width + 4) && yp < (height + 4))
    irows[BW * ty + tx] = d_Data[yl * pitch + xl];
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if (yp < (height + 4) && tx < W2)
  {
    float *ptr = &irows[BW * ty + 2 * tx];
    brows[W2 * ty + tx] = k0 * (ptr[0] + ptr[4]) + k1 * (ptr[1] + ptr[3]) + k2 * ptr[2];
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  const int xs = item_ct1.get_group(2) * W2 + tx;
  const int ys = item_ct1.get_group(1) * H2 + ty;
  if (tx < W2 && ty < H2 && xs < (width / 2) && ys < (height / 2))
  {
    float *ptr = &brows[W2 * (ty * 2) + tx];
    d_Result[ys * newpitch + xs] = k0 * (ptr[0] + ptr[4 * W2]) + k1 * (ptr[1 * W2] + ptr[3 * W2]) + k2 * ptr[2 * W2];
  }
}

void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch,
               const sycl::nd_item<3> &item_ct1, float const *d_ScaleDownKernel,
               float *inrow, float *brow, int *yRead, int *yWrite)
{

#define dx2 (SCALEDOWN_W / 2)
  const int tx = item_ct1.get_local_id(2);
  const int tx0 = tx + 0 * dx2;
  const int tx1 = tx + 1 * dx2;
  const int tx2 = tx + 2 * dx2;
  const int tx3 = tx + 3 * dx2;
  const int tx4 = tx + 4 * dx2;
  const int xStart = item_ct1.get_group(2) * SCALEDOWN_W;
  const int yStart = item_ct1.get_group(1) * SCALEDOWN_H;
  const int xWrite = xStart / 2 + tx;
  float k0 = d_ScaleDownKernel[0];
  float k1 = d_ScaleDownKernel[1];
  float k2 = d_ScaleDownKernel[2];
  if (tx < SCALEDOWN_H + 4)
  {
    int y = yStart + tx - 2;
    y = (y < 0 ? 0 : y);
    y = (y >= height ? height - 1 : y);
    yRead[tx] = y * pitch;
    yWrite[tx] = (yStart + tx - 4) / 2 * newpitch;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  int xRead = xStart + tx - 2;
  xRead = (xRead < 0 ? 0 : xRead);
  xRead = (xRead >= width ? width - 1 : xRead);

  int maxtx = sycl::min(dx2, width / 2 - xStart / 2);
  for (int dy = 0; dy < SCALEDOWN_H + 4; dy += 5)
  {
    {
      inrow[tx] = d_Data[yRead[dy + 0] + xRead];
      /*
      DPCT1118:3: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      if (tx < maxtx)
      {
        brow[tx4] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 4 && !(dy & 1))
          d_Result[yWrite[dy + 0] + xWrite] = k2 * brow[tx2] + k0 * (brow[tx0] + brow[tx4]) + k1 * (brow[tx1] + brow[tx3]);
      }
      /*
      DPCT1118:4: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (dy < (SCALEDOWN_H + 3))
    {
      inrow[tx] = d_Data[yRead[dy + 1] + xRead];
      /*
      DPCT1118:5: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      if (tx < maxtx)
      {
        brow[tx0] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 3 && (dy & 1))
          d_Result[yWrite[dy + 1] + xWrite] = k2 * brow[tx3] + k0 * (brow[tx1] + brow[tx0]) + k1 * (brow[tx2] + brow[tx4]);
      }
      /*
      DPCT1118:6: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (dy < (SCALEDOWN_H + 2))
    {
      inrow[tx] = d_Data[yRead[dy + 2] + xRead];
      /*
      DPCT1118:7: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      if (tx < maxtx)
      {
        brow[tx1] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 2 && !(dy & 1))
          d_Result[yWrite[dy + 2] + xWrite] = k2 * brow[tx4] + k0 * (brow[tx2] + brow[tx1]) + k1 * (brow[tx3] + brow[tx0]);
      }
      /*
      DPCT1118:8: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (dy < (SCALEDOWN_H + 1))
    {
      inrow[tx] = d_Data[yRead[dy + 3] + xRead];
      /*
      DPCT1118:9: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      if (tx < maxtx)
      {
        brow[tx2] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 1 && (dy & 1))
          d_Result[yWrite[dy + 3] + xWrite] = k2 * brow[tx0] + k0 * (brow[tx3] + brow[tx2]) + k1 * (brow[tx4] + brow[tx1]);
      }
      /*
      DPCT1118:10: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (dy < SCALEDOWN_H)
    {
      inrow[tx] = d_Data[yRead[dy + 4] + xRead];
      /*
      DPCT1118:11: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      if (tx < dx2 && xWrite < width / 2)
      {
        brow[tx3] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (!(dy & 1))
          d_Result[yWrite[dy + 4] + xWrite] = k2 * brow[tx1] + k0 * (brow[tx4] + brow[tx3]) + k1 * (brow[tx0] + brow[tx2]);
      }
      /*
      DPCT1118:12: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
    }
  }
}

void ScaleUp(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch,
             const sycl::nd_item<3> &item_ct1)
{
  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  int x = item_ct1.get_group(2) * SCALEUP_W + 2 * tx;
  int y = item_ct1.get_group(1) * SCALEUP_H + 2 * ty;
  if (x < 2 * width && y < 2 * height)
  {
    int xl = item_ct1.get_group(2) * (SCALEUP_W / 2) + tx;
    int yu = item_ct1.get_group(1) * (SCALEUP_H / 2) + ty;
    int xr = sycl::min(xl + 1, width - 1);
    int yd = sycl::min(yu + 1, height - 1);
    float vul = d_Data[yu * pitch + xl];
    float vur = d_Data[yu * pitch + xr];
    float vdl = d_Data[yd * pitch + xl];
    float vdr = d_Data[yd * pitch + xr];
    d_Result[(y + 0) * newpitch + x + 0] = vul;
    d_Result[(y + 0) * newpitch + x + 1] = 0.50f * (vul + vur);
    d_Result[(y + 1) * newpitch + x + 0] = 0.50f * (vul + vdl);
    d_Result[(y + 1) * newpitch + x + 1] = 0.25f * (vul + vur + vdl + vdr);
  }
}

/*
DPCT1110:13: The total declared local variable size in device function
ExtractSiftDescriptors exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void ExtractSiftDescriptors(dpct::image_accessor_ext<float, 2> texObj,
                            SiftPoint *d_sift, int fstPts, float subsampling,
                            const sycl::nd_item<3> &item_ct1, float *gauss,
                            float *buffer, float *sums)
{

  const int tx = item_ct1.get_local_id(2); // 0 -> 16
  const int ty = item_ct1.get_local_id(1); // 0 -> 8
  const int idx = ty * 16 + tx;
  const int bx = item_ct1.get_group(2) + fstPts; // 0 -> numPts
  if (ty == 0)
    gauss[tx] = sycl::exp(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);
  buffer[idx] = 0.0f;
  /*
  DPCT1065:92: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Compute angles and gradients
  float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
  float sina = sycl::sin(theta); // cosa -sina
  float cosa = sycl::cos(theta); // sina  cosa
  float scale = 12.0f / 16.0f * d_sift[bx].scale;
  float ssina = scale * sina;
  float scosa = scale * cosa;

  for (int y = ty; y < 16; y += 8)
  {
    float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
    float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
    float dx = texObj.read(xpos + cosa, ypos + sina) -
               texObj.read(xpos - cosa, ypos - sina);
    float dy = texObj.read(xpos - sina, ypos + cosa) -
               texObj.read(xpos + sina, ypos - cosa);
    float grad = gauss[y] * gauss[tx] * sycl::sqrt(dx * dx + dy * dy);
    float angf = 4.0f / 3.1415f * sycl::atan2(dy, dx) + 4.0f;

    int hori = (tx + 2) / 4 - 1; // Convert from (tx,y,angle) to bins
    float horf = (tx - 1.5f) / 4.0f - hori;
    float ihorf = 1.0f - horf;
    int veri = (y + 2) / 4 - 1;
    float verf = (y - 1.5f) / 4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi < 7 ? angi + 1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;

    int hist = 8 * (4 * veri + hori); // Each gradient measure is interpolated
    int p1 = angi + hist;             // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx >= 2)
    {
      float grad1 = ihorf * grad;
      if (y >= 2)
      { // Upper left
        float grad2 = iverf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2, angf * grad2);
      }
      if (y <= 13)
      { // Lower left
        float grad2 = verf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 32, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 32, angf * grad2);
      }
    }
    if (tx <= 13)
    {
      float grad1 = horf * grad;
      if (y >= 2)
      { // Upper right
        float grad2 = iverf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 8, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 8, angf * grad2);
      }
      if (y <= 13)
      { // Lower right
        float grad2 = verf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 40, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 40, angf * grad2);
      }
    }
  }
  /*
  DPCT1065:93: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Normalize twice and suppress peaks first time
  float sum = buffer[idx] * buffer[idx];
  for (int i = 16; i > 0; i /= 2)
    sum += ShiftDown(sum, i, item_ct1);
  if ((idx & 31) == 0)
    sums[idx / 32] = sum;
  /*
  DPCT1065:94: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
  tsum1 = sycl::min(buffer[idx] * sycl::rsqrt(tsum1), 0.2f);

  sum = tsum1 * tsum1;
  for (int i = 16; i > 0; i /= 2)
    sum += ShiftDown(sum, i, item_ct1);
  if ((idx & 31) == 0)
    sums[idx / 32] = sum;
  /*
  DPCT1065:95: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
  float *desc = d_sift[bx].data;
  desc[idx] = tsum1 * sycl::rsqrt(tsum2);
  if (idx == 0)
  {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}

float FastAtan2(float y, float x)
{
  float absx = sycl::fabs(x);
  float absy = sycl::fabs(y);
  /*
  DPCT1013:96: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  float a = sycl::min(absx, absy) / sycl::max(absx, absy);
  float s = a * a;
  float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
  r = (absy > absx ? 1.57079637f - r : r);
  r = (x < 0 ? 3.14159274f - r : r);
  r = (y < 0 ? -r : r);
  return r;
}

// __global__ void ExtractSiftDescriptorsCONSTNew(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave)
/*
DPCT1110:14: The total declared local variable size in device function
ExtractSiftDescriptorsCONSTNew exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void ExtractSiftDescriptorsCONSTNew(float *texObj, int pitch, SiftPoint *d_sift,
                                    float subsampling, int octave,
                                    const sycl::nd_item<3> &item_ct1,
                                    int d_MaxNumPoints,
                                    unsigned int *d_PointCounter, float *gauss,
                                    float *buffer, float *sums)
{

  const int tx = item_ct1.get_local_id(2); // 0 -> 16
  const int ty = item_ct1.get_local_id(1); // 0 -> 8
  const int idx = ty * 16 + tx;
  if (ty == 0)
    gauss[tx] = sycl::native::exp(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);

  int fstPts = dpct::min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = dpct::min(d_PointCounter[2 * octave + 1], d_MaxNumPoints);
  // if (tx==0 && ty==0)
  //   printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts);
  for (int bx = item_ct1.get_group(2) + fstPts; bx < totPts;
       bx += item_ct1.get_group_range(2))
  {

    buffer[idx] = 0.0f;
    /*
    DPCT1118:15: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Compute angles and gradients
    float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
    float sina = sycl::sin(theta); // cosa -sina
    float cosa = sycl::cos(theta); // sina  cosa
    float scale = 12.0f / 16.0f * d_sift[bx].scale;
    float ssina = scale * sina;
    float scosa = scale * cosa;

    for (int y = ty; y < 16; y += 8)
    {
      float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
      float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;

      // float dx = tex2D<float>(texObj, xpos + cosa, ypos + sina) -
      //            tex2D<float>(texObj, xpos - cosa, ypos - sina);
      // float dy = tex2D<float>(texObj, xpos - sina, ypos + cosa) -
      //            tex2D<float>(texObj, xpos + sina, ypos - cosa);

      int xi1 = xpos + cosa;
      int yi1 = ypos + sina;

      int xi2 = xpos - cosa;
      int yi2 = ypos - sina;

      float dx = *(texObj + yi1 * pitch + xi1) -
                 *(texObj + yi2 * pitch + xi2);

      xi1 = xpos - sina;
      yi1 = ypos + cosa;

      xi2 = xpos + sina;
      yi2 = ypos - cosa;

      float dy = *(texObj + yi1 * pitch + xi1) -
                 *(texObj + yi2 * pitch + xi2);

      /*
      DPCT1013:102: The rounding mode could not be specified and the generated
      code may have different accuracy than the original code. Verify the
      correctness. SYCL math built-in function rounding mode is aligned with
      OpenCL C 1.2 standard.
      */
      float grad = gauss[y] * gauss[tx] * sycl::sqrt(dx * dx + dy * dy);
      float angf = 4.0f / 3.1415f * FastAtan2(dy, dx) + 4.0f;

      int hori = (tx + 2) / 4 - 1; // Convert from (tx,y,angle) to bins
      float horf = (tx - 1.5f) / 4.0f - hori;
      float ihorf = 1.0f - horf;
      int veri = (y + 2) / 4 - 1;
      float verf = (y - 1.5f) / 4.0f - veri;
      float iverf = 1.0f - verf;
      int angi = angf;
      int angp = (angi < 7 ? angi + 1 : 0);
      angf -= angi;
      float iangf = 1.0f - angf;

      int hist = 8 * (4 * veri + hori); // Each gradient measure is interpolated
      int p1 = angi + hist;             // in angles, xpos and ypos -> 8 stores
      int p2 = angp + hist;
      if (tx >= 2)
      {
        float grad1 = ihorf * grad;
        if (y >= 2)
        { // Upper left
          float grad2 = iverf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2, angf * grad2);
        }
        if (y <= 13)
        { // Lower left
          float grad2 = verf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1 + 32, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2 + 32, angf * grad2);
        }
      }
      if (tx <= 13)
      {
        float grad1 = horf * grad;
        if (y >= 2)
        { // Upper right
          float grad2 = iverf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1 + 8, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2 + 8, angf * grad2);
        }
        if (y <= 13)
        { // Lower right
          float grad2 = verf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1 + 40, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2 + 40, angf * grad2);
        }
      }
    }
    /*
    DPCT1118:16: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:98: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Normalize twice and suppress peaks first time
    float sum = buffer[idx] * buffer[idx];
    for (int i = 16; i > 0; i /= 2)
      sum += ShiftDown(sum, i, item_ct1);
    if ((idx & 31) == 0)
      sums[idx / 32] = sum;
    /*
    DPCT1118:17: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:99: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
    tsum1 = sycl::min(buffer[idx] * sycl::rsqrt(tsum1), 0.2f);

    sum = tsum1 * tsum1;
    for (int i = 16; i > 0; i /= 2)
      sum += ShiftDown(sum, i, item_ct1);
    if ((idx & 31) == 0)
      sums[idx / 32] = sum;
    /*
    DPCT1118:18: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:100: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * sycl::rsqrt(tsum2);
    if (idx == 0)
    {
      d_sift[bx].xpos *= subsampling;
      d_sift[bx].ypos *= subsampling;
      d_sift[bx].scale *= subsampling;
    }
    /*
    DPCT1118:19: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:101: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

/*
DPCT1110:20: The total declared local variable size in device function
ExtractSiftDescriptorsCONST exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void ExtractSiftDescriptorsCONST(dpct::image_accessor_ext<float, 2> texObj,
                                 SiftPoint *d_sift, float subsampling,
                                 int octave, const sycl::nd_item<3> &item_ct1,
                                 int d_MaxNumPoints,
                                 unsigned int *d_PointCounter, float *gauss,
                                 float *buffer, float *sums)
{

  const int tx = item_ct1.get_local_id(2); // 0 -> 16
  const int ty = item_ct1.get_local_id(1); // 0 -> 8
  const int idx = ty * 16 + tx;
  if (ty == 0)
    gauss[tx] = sycl::exp(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);

  int fstPts = dpct::min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = dpct::min(d_PointCounter[2 * octave + 1], d_MaxNumPoints);
  // if (tx==0 && ty==0)
  //   printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts);
  for (int bx = item_ct1.get_group(2) + fstPts; bx < totPts;
       bx += item_ct1.get_group_range(2))
  {

    buffer[idx] = 0.0f;
    /*
    DPCT1118:21: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:103: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Compute angles and gradients
    float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
    float sina = sycl::sin(theta); // cosa -sina
    float cosa = sycl::cos(theta); // sina  cosa
    float scale = 12.0f / 16.0f * d_sift[bx].scale;
    float ssina = scale * sina;
    float scosa = scale * cosa;

    for (int y = ty; y < 16; y += 8)
    {
      float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
      float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
      float dx = texObj.read(xpos + cosa, ypos + sina) -
                 texObj.read(xpos - cosa, ypos - sina);
      float dy = texObj.read(xpos - sina, ypos + cosa) -
                 texObj.read(xpos + sina, ypos - cosa);
      float grad = gauss[y] * gauss[tx] * sycl::sqrt(dx * dx + dy * dy);
      float angf = 4.0f / 3.1415f * sycl::atan2(dy, dx) + 4.0f;

      int hori = (tx + 2) / 4 - 1; // Convert from (tx,y,angle) to bins
      float horf = (tx - 1.5f) / 4.0f - hori;
      float ihorf = 1.0f - horf;
      int veri = (y + 2) / 4 - 1;
      float verf = (y - 1.5f) / 4.0f - veri;
      float iverf = 1.0f - verf;
      int angi = angf;
      int angp = (angi < 7 ? angi + 1 : 0);
      angf -= angi;
      float iangf = 1.0f - angf;

      int hist = 8 * (4 * veri + hori); // Each gradient measure is interpolated
      int p1 = angi + hist;             // in angles, xpos and ypos -> 8 stores
      int p2 = angp + hist;
      if (tx >= 2)
      {
        float grad1 = ihorf * grad;
        if (y >= 2)
        { // Upper left
          float grad2 = iverf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2, angf * grad2);
        }
        if (y <= 13)
        { // Lower left
          float grad2 = verf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1 + 32, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2 + 32, angf * grad2);
        }
      }
      if (tx <= 13)
      {
        float grad1 = horf * grad;
        if (y >= 2)
        { // Upper right
          float grad2 = iverf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1 + 8, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2 + 8, angf * grad2);
        }
        if (y <= 13)
        { // Lower right
          float grad2 = verf * grad1;
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p1 + 40, iangf * grad2);
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              buffer + p2 + 40, angf * grad2);
        }
      }
    }
    /*
    DPCT1118:22: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:104: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Normalize twice and suppress peaks first time
    float sum = buffer[idx] * buffer[idx];
    for (int i = 16; i > 0; i /= 2)
      sum += ShiftDown(sum, i, item_ct1);
    if ((idx & 31) == 0)
      sums[idx / 32] = sum;
    /*
    DPCT1118:23: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:105: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
    tsum1 = sycl::min(buffer[idx] * sycl::rsqrt(tsum1), 0.2f);

    sum = tsum1 * tsum1;
    for (int i = 16; i > 0; i /= 2)
      sum += ShiftDown(sum, i, item_ct1);
    if ((idx & 31) == 0)
      sums[idx / 32] = sum;
    /*
    DPCT1118:24: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:106: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * sycl::rsqrt(tsum2);
    if (idx == 0)
    {
      d_sift[bx].xpos *= subsampling;
      d_sift[bx].ypos *= subsampling;
      d_sift[bx].scale *= subsampling;
    }
    /*
    DPCT1118:25: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:107: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

/*
DPCT1110:26: The total declared local variable size in device function
ExtractSiftDescriptorsOld exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void ExtractSiftDescriptorsOld(dpct::image_accessor_ext<float, 2> texObj,
                               SiftPoint *d_sift, int fstPts, float subsampling,
                               const sycl::nd_item<3> &item_ct1, float *gauss,
                               float *buffer, float *sums)
{

  const int tx = item_ct1.get_local_id(2); // 0 -> 16
  const int ty = item_ct1.get_local_id(1); // 0 -> 8
  const int idx = ty * 16 + tx;
  const int bx = item_ct1.get_group(2) + fstPts; // 0 -> numPts
  if (ty == 0)
    gauss[tx] = sycl::exp(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);
  buffer[idx] = 0.0f;
  /*
  DPCT1065:108: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Compute angles and gradients
  float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
  float sina = sycl::sin(theta); // cosa -sina
  float cosa = sycl::cos(theta); // sina  cosa
  float scale = 12.0f / 16.0f * d_sift[bx].scale;
  float ssina = scale * sina;
  float scosa = scale * cosa;

  for (int y = ty; y < 16; y += 8)
  {
    float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
    float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
    float dx = texObj.read(xpos + cosa, ypos + sina) -
               texObj.read(xpos - cosa, ypos - sina);
    float dy = texObj.read(xpos - sina, ypos + cosa) -
               texObj.read(xpos + sina, ypos - cosa);
    float grad = gauss[y] * gauss[tx] * sycl::sqrt(dx * dx + dy * dy);
    float angf = 4.0f / 3.1415f * sycl::atan2(dy, dx) + 4.0f;

    int hori = (tx + 2) / 4 - 1; // Convert from (tx,y,angle) to bins
    float horf = (tx - 1.5f) / 4.0f - hori;
    float ihorf = 1.0f - horf;
    int veri = (y + 2) / 4 - 1;
    float verf = (y - 1.5f) / 4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi < 7 ? angi + 1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;

    int hist = 8 * (4 * veri + hori); // Each gradient measure is interpolated
    int p1 = angi + hist;             // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx >= 2)
    {
      float grad1 = ihorf * grad;
      if (y >= 2)
      { // Upper left
        float grad2 = iverf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2, angf * grad2);
      }
      if (y <= 13)
      { // Lower left
        float grad2 = verf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 32, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 32, angf * grad2);
      }
    }
    if (tx <= 13)
    {
      float grad1 = horf * grad;
      if (y >= 2)
      { // Upper right
        float grad2 = iverf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 8, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 8, angf * grad2);
      }
      if (y <= 13)
      { // Lower right
        float grad2 = verf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 40, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 40, angf * grad2);
      }
    }
  }
  /*
  DPCT1065:109: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Normalize twice and suppress peaks first time
  if (idx < 64)
    sums[idx] = buffer[idx] * buffer[idx] + buffer[idx + 64] * buffer[idx + 64];
  /*
  DPCT1065:110: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 32)
    sums[idx] = sums[idx] + sums[idx + 32];
  /*
  DPCT1065:111: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 16)
    sums[idx] = sums[idx] + sums[idx + 16];
  /*
  DPCT1065:112: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 8)
    sums[idx] = sums[idx] + sums[idx + 8];
  /*
  DPCT1065:113: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 4)
    sums[idx] = sums[idx] + sums[idx + 4];
  /*
  DPCT1065:114: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
  buffer[idx] = buffer[idx] * sycl::rsqrt(tsum1);

  if (buffer[idx] > 0.2f)
    buffer[idx] = 0.2f;
  /*
  DPCT1065:115: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 64)
    sums[idx] = buffer[idx] * buffer[idx] + buffer[idx + 64] * buffer[idx + 64];
  /*
  DPCT1065:116: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 32)
    sums[idx] = sums[idx] + sums[idx + 32];
  /*
  DPCT1065:117: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 16)
    sums[idx] = sums[idx] + sums[idx + 16];
  /*
  DPCT1065:118: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 8)
    sums[idx] = sums[idx] + sums[idx + 8];
  /*
  DPCT1065:119: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (idx < 4)
    sums[idx] = sums[idx] + sums[idx + 4];
  /*
  DPCT1065:120: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];

  float *desc = d_sift[bx].data;
  desc[idx] = buffer[idx] * sycl::rsqrt(tsum2);
  if (idx == 0)
  {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}

/*
DPCT1110:27: The total declared local variable size in device function
ExtractSiftDescriptor exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void ExtractSiftDescriptor(dpct::image_accessor_ext<float, 2> texObj,
                           SiftPoint *d_sift, float subsampling, int octave,
                           int bx, const sycl::nd_item<3> &item_ct1,
                           float *gauss, float *buffer, float *sums)
{

  const int idx = item_ct1.get_local_id(2);
  const int tx = idx & 15; // 0 -> 16
  const int ty = idx / 16; // 0 -> 8
  if (ty == 0)
    gauss[tx] = sycl::exp(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);
  buffer[idx] = 0.0f;
  /*
  DPCT1065:121: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Compute angles and gradients
  float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
  float sina = sycl::sin(theta); // cosa -sina
  float cosa = sycl::cos(theta); // sina  cosa
  float scale = 12.0f / 16.0f * d_sift[bx].scale;
  float ssina = scale * sina;
  float scosa = scale * cosa;

  for (int y = ty; y < 16; y += 8)
  {
    float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
    float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
    float dx = texObj.read(xpos + cosa, ypos + sina) -
               texObj.read(xpos - cosa, ypos - sina);
    float dy = texObj.read(xpos - sina, ypos + cosa) -
               texObj.read(xpos + sina, ypos - cosa);
    float grad = gauss[y] * gauss[tx] * sycl::sqrt(dx * dx + dy * dy);
    float angf = 4.0f / 3.1415f * sycl::atan2(dy, dx) + 4.0f;

    int hori = (tx + 2) / 4 - 1; // Convert from (tx,y,angle) to bins
    float horf = (tx - 1.5f) / 4.0f - hori;
    float ihorf = 1.0f - horf;
    int veri = (y + 2) / 4 - 1;
    float verf = (y - 1.5f) / 4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi < 7 ? angi + 1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;

    int hist = 8 * (4 * veri + hori); // Each gradient measure is interpolated
    int p1 = angi + hist;             // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx >= 2)
    {
      float grad1 = ihorf * grad;
      if (y >= 2)
      { // Upper left
        float grad2 = iverf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2, angf * grad2);
      }
      if (y <= 13)
      { // Lower left
        float grad2 = verf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 32, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 32, angf * grad2);
      }
    }
    if (tx <= 13)
    {
      float grad1 = horf * grad;
      if (y >= 2)
      { // Upper right
        float grad2 = iverf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 8, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 8, angf * grad2);
      }
      if (y <= 13)
      { // Lower right
        float grad2 = verf * grad1;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p1 + 40, iangf * grad2);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            buffer + p2 + 40, angf * grad2);
      }
    }
  }
  /*
  DPCT1065:122: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Normalize twice and suppress peaks first time
  float sum = buffer[idx] * buffer[idx];
  for (int i = 16; i > 0; i /= 2)
    sum += ShiftDown(sum, i, item_ct1);
  if ((idx & 31) == 0)
    sums[idx / 32] = sum;
  /*
  DPCT1065:123: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
  tsum1 = sycl::min(buffer[idx] * sycl::rsqrt(tsum1), 0.2f);

  sum = tsum1 * tsum1;
  for (int i = 16; i > 0; i /= 2)
    sum += ShiftDown(sum, i, item_ct1);
  if ((idx & 31) == 0)
    sums[idx / 32] = sum;
  /*
  DPCT1065:124: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
  float *desc = d_sift[bx].data;
  desc[idx] = tsum1 * sycl::rsqrt(tsum2);
  if (idx == 0)
  {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
  /*
  DPCT1065:125: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
}

void RescalePositions(SiftPoint *d_sift, int numPts, float scale,
                      const sycl::nd_item<3> &item_ct1)
{
  int num = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (num < numPts)
  {
    d_sift[num].xpos *= scale;
    d_sift[num].ypos *= scale;
    d_sift[num].scale *= scale;
  }
}

void ComputeOrientations(dpct::image_accessor_ext<float, 2> texObj,
                         SiftPoint *d_Sift, int fstPts,
                         const sycl::nd_item<3> &item_ct1, int d_MaxNumPoints,
                         unsigned int *d_PointCounter, float *hist,
                         float *gauss)
{

  const int tx = item_ct1.get_local_id(2);
  const int bx = item_ct1.get_group(2) + fstPts;
  float i2sigma2 = -1.0f / (4.5f * d_Sift[bx].scale * d_Sift[bx].scale);
  if (tx < 11)
    gauss[tx] = sycl::exp(i2sigma2 * (tx - 5) * (tx - 5));
  if (tx < 64)
    hist[tx] = 0.0f;
  /*
  DPCT1065:126: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  float xp = d_Sift[bx].xpos - 4.5f;
  float yp = d_Sift[bx].ypos - 4.5f;
  int yd = tx / 11;
  int xd = tx - yd * 11;
  float xf = xp + xd;
  float yf = yp + yd;
  if (yd < 11)
  {
    float dx = texObj.read(xf + 1.0, yf) - texObj.read(xf - 1.0, yf);
    float dy = texObj.read(xf, yf + 1.0) - texObj.read(xf, yf - 1.0);
    int bin = 16.0f * sycl::atan2(dy, dx) / 3.1416f + 16.5f;
    if (bin > 31)
      bin = 0;
    float grad = sycl::sqrt(dx * dx + dy * dy);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &hist[bin], grad * gauss[xd] * gauss[yd]);
  }
  /*
  DPCT1065:127: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  int x1m = (tx >= 1 ? tx - 1 : tx + 31);
  int x1p = (tx <= 30 ? tx + 1 : tx - 31);
  if (tx < 32)
  {
    int x2m = (tx >= 2 ? tx - 2 : tx + 30);
    int x2p = (tx <= 29 ? tx + 2 : tx - 30);
    hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
  }
  /*
  DPCT1065:128: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx < 32)
  {
    float v = hist[32 + tx];
    if(x1p < 32 && x1m < 32)
      hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
  }
  /*
  DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tx == 0)
  {
    float maxval1 = 0.0;
    float maxval2 = 0.0;
    int i1 = -1;
    int i2 = -1;
    for (int i = 0; i < 32; i++)
    {
      float v = hist[i];
      if (v > maxval1)
      {
        maxval2 = maxval1;
        maxval1 = v;
        i2 = i1;
        i1 = i;
      }
      else if (v > maxval2)
      {
        maxval2 = v;
        i2 = i;
      }
    }
    float val1 = hist[32 + ((i1 + 1) & 31)];
    float val2 = hist[32 + ((i1 + 31) & 31)];
    float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
    d_Sift[bx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
    if (maxval2 > 0.8f * maxval1)
    {
      float val1 = hist[32 + ((i2 + 1) & 31)];
      float val2 = hist[32 + ((i2 + 31) & 31)];
      float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
      unsigned int idx = dpct::atomic_fetch_compare_inc<
          sycl::access::address_space::generic_space>(d_PointCounter,
                                                      0x7fffffff);
      if (idx < d_MaxNumPoints)
      {
        d_Sift[idx].xpos = d_Sift[bx].xpos;
        d_Sift[idx].ypos = d_Sift[bx].ypos;
        d_Sift[idx].scale = d_Sift[bx].scale;
        d_Sift[idx].sharpness = d_Sift[bx].sharpness;
        d_Sift[idx].edgeness = d_Sift[bx].edgeness;
        d_Sift[idx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
        ;
        d_Sift[idx].subsampling = d_Sift[bx].subsampling;
      }
    }
  }
}

// With constant number of blocks
/*
DPCT1110:28: The total declared local variable size in device function
ComputeOrientationsCONSTNew exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void ComputeOrientationsCONSTNew(float *image, int w, int p, int h,
                                 SiftPoint *d_Sift, int octave,
                                 const sycl::nd_item<3> &item_ct1,
                                 int d_MaxNumPoints,
                                 unsigned int *d_PointCounter,
                                 sycl::local_accessor<float, 2> img,
                                 sycl::local_accessor<float, 2> tmp,
                                 float *hist, float *gaussx, float *gaussy)
{
#define RAD 9
#define WID (2 * RAD + 1)
#define LEN 32 //%%%% Note: Lowe suggests 36, not 32

  const int tx = item_ct1.get_local_id(2);

  int fstPts = dpct::min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = dpct::min(d_PointCounter[2 * octave + 0], d_MaxNumPoints);
  for (int bx = item_ct1.get_group(2) + fstPts; bx < totPts;
       bx += item_ct1.get_group_range(2))
  {

    float sc = d_Sift[bx].scale;
    for (int i = tx; i < 2 * LEN; i += item_ct1.get_local_range(2))
      hist[i] = 0.0f;
    float xp = d_Sift[bx].xpos;
    float yp = d_Sift[bx].ypos;
    int xi = (int)xp;
    int yi = (int)yp;
    float xf = xp - xi;
    float yf = yp - yi;
    for (int i = tx; i < WID * WID; i += item_ct1.get_local_range(2))
    {
      int y = i / WID;
      int x = i - y * WID;
      int xp = sycl::max(sycl::min(x - RAD + xi, w - 1), 0);
      int yp = sycl::max(sycl::min(y - RAD + yi, h - 1), 0);
      img[y][x] = image[yp * p + xp];
    }
    float fac[5];
    fac[1] = fac[3] =
        (sc > 0.5f ? sycl::native::exp(-1.0f / (2.0f * (sc * sc - 0.25f)))
                   : 0.0f);
    fac[0] = fac[4] =
        (sc > 0.5f ? sycl::native::exp(-4.0f / (2.0f * (sc * sc - 0.25f)))
                   : 0.0f);
    fac[2] = 1.0f;
    float i2sigma2 = -1.0f / (2.0f * 2.0f * 2.0f * sc * sc); //%%%% Note: Lowe suggests 1.5, not 2.0
    if (tx < WID)
    {
      gaussx[tx] =
          sycl::native::exp(i2sigma2 * (tx - RAD - xf) * (tx - RAD - xf));
      gaussy[tx] =
          sycl::native::exp(i2sigma2 * (tx - RAD - yf) * (tx - RAD - yf));
    }
    /*
    DPCT1118:29: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:130: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = tx; i < (WID - 4) * WID; i += item_ct1.get_local_range(2))
    {
      int y = i / WID;
      int x = i - y * WID;
      y += 2;
      tmp[y][x] = img[y][x] + fac[1] * (img[y - 1][x] + img[y + 1][x]) +
                  fac[0] * (img[y - 2][x] + img[y + 2][x]);
    }
    /*
    DPCT1118:30: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:131: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = tx; i < (WID - 4) * (WID - 4);
         i += item_ct1.get_local_range(2))
    {
      int y = i / (WID - 4);
      int x = i - y * (WID - 4);
      x += 2;
      y += 2;
      img[y][x] = tmp[y][x] + fac[1] * (tmp[y][x - 1] + tmp[y][x + 1]) +
                  fac[0] * (tmp[y][x - 2] + tmp[y][x + 2]);
    }
    /*
    DPCT1118:31: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:132: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = tx; i < (WID - 6) * (WID - 6);
         i += item_ct1.get_local_range(2))
    {
      int y = i / (WID - 6);
      int x = i - y * (WID - 6);
      x += 3;
      y += 3;
      float dx = img[y][x + 1] - img[y][x - 1];
      float dy = img[y + 1][x] - img[y - 1][x];
      int bin =
          (int)((LEN / 2) * sycl::atan2(dy, dx) / 3.1416f + (LEN / 2) + 0.5f) %
          LEN;
      /*
      DPCT1013:135: The rounding mode could not be specified and the generated
      code may have different accuracy than the original code. Verify the
      correctness. SYCL math built-in function rounding mode is aligned with
      OpenCL C 1.2 standard.
      */
      float grad = sycl::sqrt(dx * dx + dy * dy);
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &hist[LEN + bin], grad * gaussx[x] * gaussy[y]);
    }
    /*
    DPCT1118:32: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:133: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int x1m = (tx >= 1 ? tx - 1 : tx + LEN - 1);
    int x1p = (tx < (LEN - 1) ? tx + 1 : tx - LEN + 1);
    int x2m = (tx >= 2 ? tx - 2 : tx + LEN - 2);
    int x2p = (tx < (LEN - 2) ? tx + 2 : tx - LEN + 2);
    if (tx < LEN)
    {
      hist[tx] = 6.0f * hist[tx + LEN] + 4.0f * (hist[x1m + LEN] + hist[x1p + LEN]) +
                 1.0f * (hist[x2m + LEN] + hist[x2p + LEN]);
      hist[tx + LEN] = 8.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) +
                       0.0f * (hist[x2m] + hist[x2p]);
      float val = hist[tx + LEN];
      hist[tx] = (val > hist[x1m + LEN] && val >= hist[x1p + LEN] ? val : 0.0f);
    }
    /*
    DPCT1118:33: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:134: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tx == 0)
    {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i = 0; i < LEN; i++)
      {
        float v = hist[i];
        if (v > maxval1)
        {
          maxval2 = maxval1;
          maxval1 = v;
          i2 = i1;
          i1 = i;
        }
        else if (v > maxval2)
        {
          maxval2 = v;
          i2 = i;
        }
      }
      float val1 = hist[LEN + ((i1 + 1) % LEN)];
      float val2 = hist[LEN + ((i1 + LEN - 1) % LEN)];
      float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
      d_Sift[bx].orientation = 360.0f * (peak < 0.0f ? peak + LEN : peak) / LEN;
      dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
          &d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave + 0]);
      if (maxval2 > 0.8f * maxval1 && true)
      {
        float val1 = hist[LEN + ((i2 + 1) % LEN)];
        float val2 = hist[LEN + ((i2 + LEN - 1) % LEN)];
        float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
        unsigned int idx = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            &d_PointCounter[2 * octave + 1], 0x7fffffff);
        if (idx < d_MaxNumPoints)
        {
          d_Sift[idx].xpos = d_Sift[bx].xpos;
          d_Sift[idx].ypos = d_Sift[bx].ypos;
          d_Sift[idx].scale = sc;
          d_Sift[idx].sharpness = d_Sift[bx].sharpness;
          d_Sift[idx].edgeness = d_Sift[bx].edgeness;
          d_Sift[idx].orientation = 360.0f * (peak < 0.0f ? peak + LEN : peak) / LEN;
          d_Sift[idx].subsampling = d_Sift[bx].subsampling;
        }
      }
    }
  }
#undef RAD
#undef WID
#undef LEN
}

// With constant number of blocks
/*
DPCT1110:34: The total declared local variable size in device function
ComputeOrientationsCONST exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void ComputeOrientationsCONST(dpct::image_accessor_ext<float, 2> texObj,
                              SiftPoint *d_Sift, int octave,
                              const sycl::nd_item<3> &item_ct1,
                              int d_MaxNumPoints, unsigned int *d_PointCounter,
                              float *hist, float *gauss)
{

  const int tx = item_ct1.get_local_id(2);

  int fstPts = dpct::min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = dpct::min(d_PointCounter[2 * octave + 0], d_MaxNumPoints);
  for (int bx = item_ct1.get_group(2) + fstPts; bx < totPts;
       bx += item_ct1.get_group_range(2))
  {

    float i2sigma2 = -1.0f / (2.0f * 1.5f * 1.5f * d_Sift[bx].scale * d_Sift[bx].scale);
    if (tx < 11)
      gauss[tx] = sycl::exp(i2sigma2 * (tx - 5) * (tx - 5));
    if (tx < 64)
      hist[tx] = 0.0f;
    /*
    DPCT1118:35: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:136: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float xp = d_Sift[bx].xpos - 4.5f;
    float yp = d_Sift[bx].ypos - 4.5f;
    int yd = tx / 11;
    int xd = tx - yd * 11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd < 11)
    {
      float dx = texObj.read(xf + 1.0, yf) - texObj.read(xf - 1.0, yf);
      float dy = texObj.read(xf, yf + 1.0) - texObj.read(xf, yf - 1.0);
      int bin = 16.0f * sycl::atan2(dy, dx) / 3.1416f + 16.5f;
      if (bin > 31)
        bin = 0;
      float grad = sycl::sqrt(dx * dx + dy * dy);
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &hist[bin], grad * gauss[xd] * gauss[yd]);
    }
    /*
    DPCT1118:36: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:137: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int x1m = (tx >= 1 ? tx - 1 : tx + 31);
    int x1p = (tx <= 30 ? tx + 1 : tx - 31);
    if (tx < 32)
    {
      int x2m = (tx >= 2 ? tx - 2 : tx + 30);
      int x2p = (tx <= 29 ? tx + 2 : tx - 30);
      hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    /*
    DPCT1118:37: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:138: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tx < 32)
    {
      float v = hist[32 + tx];
      if(x1m < 32)
        hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
    }
    /*
    DPCT1118:38: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:139: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tx == 0)
    {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i = 0; i < 32; i++)
      {
        float v = hist[i];
        if (v > maxval1)
        {
          maxval2 = maxval1;
          maxval1 = v;
          i2 = i1;
          i1 = i;
        }
        else if (v > maxval2)
        {
          maxval2 = v;
          i2 = i;
        }
      }
      float val1 = hist[32 + ((i1 + 1) & 31)];
      float val2 = hist[32 + ((i1 + 31) & 31)];
      float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
      d_Sift[bx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
      dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
          &d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave + 0]);
      if (maxval2 > 0.8f * maxval1 && true)
      {
        float val1 = hist[32 + ((i2 + 1) & 31)];
        float val2 = hist[32 + ((i2 + 31) & 31)];
        float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
        unsigned int idx = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            &d_PointCounter[2 * octave + 1], 0x7fffffff);
        if (idx < d_MaxNumPoints)
        {
          d_Sift[idx].xpos = d_Sift[bx].xpos;
          d_Sift[idx].ypos = d_Sift[bx].ypos;
          d_Sift[idx].scale = d_Sift[bx].scale;
          d_Sift[idx].sharpness = d_Sift[bx].sharpness;
          d_Sift[idx].edgeness = d_Sift[bx].edgeness;
          d_Sift[idx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
          ;
          d_Sift[idx].subsampling = d_Sift[bx].subsampling;
        }
      }
    }
    /*
    DPCT1118:39: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:140: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

// With constant number of blocks
void OrientAndExtractCONST(dpct::image_accessor_ext<float, 2> texObj,
                           SiftPoint *d_Sift, float subsampling, int octave,
                           const sycl::nd_item<3> &item_ct1, int d_MaxNumPoints,
                           unsigned int *d_PointCounter, float *gauss,
                           float *buffer, float *sums, float *hist,
                           unsigned int &idx)
{

   //%%%%
  const int tx = item_ct1.get_local_id(2);

  int fstPts = dpct::min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = dpct::min(d_PointCounter[2 * octave + 0], d_MaxNumPoints);
  for (int bx = item_ct1.get_group(2) + fstPts; bx < totPts;
       bx += item_ct1.get_group_range(2))
  {

    float i2sigma2 = -1.0f / (4.5f * d_Sift[bx].scale * d_Sift[bx].scale);
    if (tx < 11)
      gauss[tx] = sycl::exp(i2sigma2 * (tx - 5) * (tx - 5));
    if (tx < 64)
      hist[tx] = 0.0f;
    /*
    DPCT1118:40: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:141: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float xp = d_Sift[bx].xpos - 4.5f;
    float yp = d_Sift[bx].ypos - 4.5f;
    int yd = tx / 11;
    int xd = tx - yd * 11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd < 11)
    {
      float dx = texObj.read(xf + 1.0, yf) - texObj.read(xf - 1.0, yf);
      float dy = texObj.read(xf, yf + 1.0) - texObj.read(xf, yf - 1.0);
      int bin = 16.0f * sycl::atan2(dy, dx) / 3.1416f + 16.5f;
      if (bin > 31)
        bin = 0;
      float grad = sycl::sqrt(dx * dx + dy * dy);
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &hist[bin], grad * gauss[xd] * gauss[yd]);
    }
    /*
    DPCT1118:41: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:142: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int x1m = (tx >= 1 ? tx - 1 : tx + 31);
    int x1p = (tx <= 30 ? tx + 1 : tx - 31);
    if (tx < 32)
    {
      int x2m = (tx >= 2 ? tx - 2 : tx + 30);
      int x2p = (tx <= 29 ? tx + 2 : tx - 30);
      hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    /*
    DPCT1118:42: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:143: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tx < 32)
    {
      float v = hist[32 + tx];
      if(x1m < 32)
        hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
    }
    /*
    DPCT1118:43: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:144: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tx == 0)
    {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i = 0; i < 32; i++)
      {
        float v = hist[i];
        if (v > maxval1)
        {
          maxval2 = maxval1;
          maxval1 = v;
          i2 = i1;
          i1 = i;
        }
        else if (v > maxval2)
        {
          maxval2 = v;
          i2 = i;
        }
      }
      float val1 = hist[32 + ((i1 + 1) & 31)];
      float val2 = hist[32 + ((i1 + 31) & 31)];
      float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
      d_Sift[bx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
      idx = 0xffffffff; //%%%%
      dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
          &d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave + 0]);
      if (maxval2 > 0.8f * maxval1)
      {
        float val1 = hist[32 + ((i2 + 1) & 31)];
        float val2 = hist[32 + ((i2 + 31) & 31)];
        float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
        idx = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            &d_PointCounter[2 * octave + 1], 0x7fffffff); //%%%%
        if (idx < d_MaxNumPoints)
        {
          d_Sift[idx].xpos = d_Sift[bx].xpos;
          d_Sift[idx].ypos = d_Sift[bx].ypos;
          d_Sift[idx].scale = d_Sift[bx].scale;
          d_Sift[idx].sharpness = d_Sift[bx].sharpness;
          d_Sift[idx].edgeness = d_Sift[bx].edgeness;
          d_Sift[idx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
          ;
          d_Sift[idx].subsampling = d_Sift[bx].subsampling;
        }
      }
    }
    /*
    DPCT1118:44: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:145: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    ExtractSiftDescriptor(texObj, d_Sift, subsampling, octave, bx, item_ct1,
                          gauss, buffer, sums);                        //%%%%
    if (idx < d_MaxNumPoints)                                          //%%%%
      ExtractSiftDescriptor(texObj, d_Sift, subsampling, octave, idx, item_ct1,
                            gauss, buffer, sums); //%%%%
  }
}

///////////////////////////////////////////////////////////////////////////////
// Subtract two images (multi-scale version)
///////////////////////////////////////////////////////////////////////////////

// __global__ void FindPointsMultiTest(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
// {
// #define MEMWID (MINMAX_W + 2)
//   __shared__ unsigned int cnt;
//   __shared__ unsigned short points[3 * MEMWID];

//   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//   {
//     atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
//     atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
//   }
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   if (tx == 0 && ty == 0)
//     cnt = 0;
//   __syncthreads();

//   int ypos = MINMAX_H * blockIdx.y + ty;
//   if (ypos >= height)
//     return;
//   int block = blockIdx.x / NUM_SCALES;
//   int scale = blockIdx.x - NUM_SCALES * block;
//   int minx = block * MINMAX_W;
//   int maxx = min(minx + MINMAX_W, width);
//   int xpos = minx + tx;
//   int size = pitch * height;
//   int ptr = size * scale + max(min(xpos - 1, width - 1), 0);

//   float maxv = fabs(d_Data0[ptr + ypos * pitch + 1 * size]);
//   maxv = fmaxf(maxv, ShiftDown(maxv, 16, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 8, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 4, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 2, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 1, MINMAX_W));

//   if (Shuffle(maxv, 0) > thresh)
//   {
//     int yptr1 = ptr + ypos * pitch;
//     int yptr0 = ptr + max(0, ypos - 1) * pitch;
//     int yptr2 = ptr + min(height - 1, ypos + 1) * pitch;
//     float d20 = d_Data0[yptr0 + 1 * size];
//     float d21 = d_Data0[yptr1 + 1 * size];
//     float d22 = d_Data0[yptr2 + 1 * size];
//     float d31 = d_Data0[yptr1 + 2 * size];
//     float d11 = d_Data0[yptr1];

//     float d10 = d_Data0[yptr0];
//     float d12 = d_Data0[yptr2];
//     float ymin1 = fminf(fminf(d10, d11), d12);
//     float ymax1 = fmaxf(fmaxf(d10, d11), d12);
//     float d30 = d_Data0[yptr0 + 2 * size];
//     float d32 = d_Data0[yptr2 + 2 * size];
//     float ymin3 = fminf(fminf(d30, d31), d32);
//     float ymax3 = fmaxf(fmaxf(d30, d31), d32);
//     float ymin2 = fminf(fminf(ymin1, fminf(fminf(d20, d22), d21)), ymin3);
//     float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d20, d22), d21)), ymax3);

//     float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
//     float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
//     if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
//     {
//       if (d21 < -thresh)
//       {
//         float minv = fminf(fminf(nmin2, ymin1), ymin3);
//         minv = fminf(fminf(minv, d20), d22);
//         if (d21 < minv)
//         {
//           int pos = atomicInc(&cnt, MEMWID - 1);
//           points[3 * pos + 0] = xpos - 1;
//           points[3 * pos + 1] = ypos;
//           points[3 * pos + 2] = scale;
//         }
//       }
//       if (d21 > thresh)
//       {
//         float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
//         maxv = fmaxf(fmaxf(maxv, d20), d22);
//         if (d21 > maxv)
//         {
//           int pos = atomicInc(&cnt, MEMWID - 1);
//           points[3 * pos + 0] = xpos - 1;
//           points[3 * pos + 1] = ypos;
//           points[3 * pos + 2] = scale;
//         }
//       }
//     }
//   }
//   __syncthreads();
//   if (ty == 0 && tx < cnt)
//   {
//     int xpos = points[3 * tx + 0];
//     int ypos = points[3 * tx + 1];
//     int scale = points[3 * tx + 2];
//     int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
//     float val = d_Data0[ptr];
//     float *data1 = &d_Data0[ptr];
//     float dxx = 2.0f * val - data1[-1] - data1[1];
//     float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
//     float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
//     float tra = dxx + dyy;
//     float det = dxx * dyy - dxy * dxy;
//     if (tra * tra < edgeLimit * det)
//     {
//       float edge = __fdividef(tra * tra, det);
//       float dx = 0.5f * (data1[1] - data1[-1]);
//       float dy = 0.5f * (data1[pitch] - data1[-pitch]);
//       float *data0 = d_Data0 + ptr - height * pitch;
//       float *data2 = d_Data0 + ptr + height * pitch;
//       float ds = 0.5f * (data0[0] - data2[0]);
//       float dss = 2.0f * val - data2[0] - data0[0];
//       float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
//       float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
//       float idxx = dyy * dss - dys * dys;
//       float idxy = dys * dxs - dxy * dss;
//       float idxs = dxy * dys - dyy * dxs;
//       float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
//       float idyy = dxx * dss - dxs * dxs;
//       float idys = dxy * dxs - dxx * dys;
//       float idss = dxx * dyy - dxy * dxy;
//       float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
//       float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
//       float pds = idet * (idxs * dx + idys * dy + idss * ds);
//       if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f)
//       {
//         pdx = __fdividef(dx, dxx);
//         pdy = __fdividef(dy, dyy);
//         pds = __fdividef(ds, dss);
//       }
//       float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
//       int maxPts = d_MaxNumPoints;
//       float sc = powf(2.0f, (float)scale / NUM_SCALES) * exp2f(pds * factor);
//       if (sc >= lowestScale)
//       {
//         unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 0], 0x7fffffff);
//         idx = (idx >= maxPts ? maxPts - 1 : idx);
//         d_Sift[idx].xpos = xpos + pdx;
//         d_Sift[idx].ypos = ypos + pdy;
//         d_Sift[idx].scale = sc;
//         d_Sift[idx].sharpness = val + dval;
//         d_Sift[idx].edgeness = edge;
//         d_Sift[idx].subsampling = subsampling;
//       }
//     }
//   }
// }

/*
DPCT1110:45: The total declared local variable size in device function
FindPointsMultiNew exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void FindPointsMultiNew(float *d_Data0, SiftPoint *d_Sift, int width, int pitch,
                        int height, float subsampling, float lowestScale,
                        float thresh, float factor, float edgeLimit, int octave,
                        const sycl::nd_item<3> &item_ct1, int d_MaxNumPoints,
                        unsigned int *d_PointCounter, unsigned short *points)
{
#define MEMWID (MINMAX_W + 2)

  if (item_ct1.get_group(2) == 0 && item_ct1.get_group(1) == 0 &&
      item_ct1.get_local_id(2) == 0)
  {
    dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
        &d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
    dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
        &d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
  }
  int tx = item_ct1.get_local_id(2);
  int block = item_ct1.get_group(2) / NUM_SCALES;
  int scale = item_ct1.get_group(2) - NUM_SCALES * block;
  int minx = block * MINMAX_W;
  int maxx = sycl::min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch * height;
  int ptr = size * scale + sycl::max(sycl::min(xpos - 1, width - 1), 0);

  int yloops = dpct::min(
      (unsigned int)(height - MINMAX_H * item_ct1.get_group(1)), MINMAX_H);
  float maxv = 0.0f;
  for (int y = 0; y < yloops; y++)
  {
    int ypos = MINMAX_H * item_ct1.get_group(1) + y;
    int yptr1 = ptr + ypos * pitch;
    float val = d_Data0[yptr1 + 1 * size];
    maxv = sycl::fmax(maxv, sycl::fabs(val));
  }
  // if (tx==0) printf("XXX1\n");
  if (!sycl::any_of_group(
          item_ct1.get_sub_group(),
          (0xffffffff &
           (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
              maxv > thresh))
    return;
  // if (tx==0) printf("XXX2\n");

  int ptbits = 0;
  for (int y = 0; y < yloops; y++)
  {

    int ypos = MINMAX_H * item_ct1.get_group(1) + y;
    int yptr1 = ptr + ypos * pitch;
    float d11 = d_Data0[yptr1 + 1 * size];
    if (sycl::any_of_group(
            item_ct1.get_sub_group(),
            (0xffffffff &
             (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                sycl::fabs(d11) > thresh))
    {

      int yptr0 = ptr + sycl::max(0, ypos - 1) * pitch;
      int yptr2 = ptr + sycl::min(height - 1, ypos + 1) * pitch;
      float d01 = d_Data0[yptr1];
      float d10 = d_Data0[yptr0 + 1 * size];
      float d12 = d_Data0[yptr2 + 1 * size];
      float d21 = d_Data0[yptr1 + 2 * size];

      float d00 = d_Data0[yptr0];
      float d02 = d_Data0[yptr2];
      float ymin1 = sycl::fmin(sycl::fmin(d00, d01), d02);
      float ymax1 = sycl::fmax(sycl::fmax(d00, d01), d02);
      float d20 = d_Data0[yptr0 + 2 * size];
      float d22 = d_Data0[yptr2 + 2 * size];
      float ymin3 = sycl::fmin(sycl::fmin(d20, d21), d22);
      float ymax3 = sycl::fmax(sycl::fmax(d20, d21), d22);
      float ymin2 = sycl::fmin(
          sycl::fmin(ymin1, sycl::fmin(sycl::fmin(d10, d12), d11)), ymin3);
      float ymax2 = sycl::fmax(
          sycl::fmax(ymax1, sycl::fmax(sycl::fmax(d10, d12), d11)), ymax3);

      float nmin2 = sycl::fmin(ShiftUp(ymin2, 1, item_ct1),
                               ShiftDown(ymin2, 1, item_ct1));
      float nmax2 = sycl::fmax(ShiftUp(ymax2, 1, item_ct1),
                               ShiftDown(ymax2, 1, item_ct1));
      float minv = sycl::fmin(sycl::fmin(nmin2, ymin1), ymin3);
      minv = sycl::fmin(sycl::fmin(minv, d10), d12);
      float maxv = sycl::fmax(sycl::fmax(nmax2, ymax1), ymax3);
      maxv = sycl::fmax(sycl::fmax(maxv, d10), d12);

      if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
        ptbits |= ((d11 < sycl::fmin(-thresh, minv)) |
                   (d11 > sycl::fmax(thresh, maxv)))
                  << y;
    }
  }

  unsigned int totbits = sycl::popcount(ptbits);
  unsigned int numbits = totbits;
  for (int d = 1; d < 32; d <<= 1)
  {
    unsigned int num = ShiftUp(totbits, d, item_ct1);
    if (tx >= d)
      totbits += num;
  }
  int pos = totbits - numbits;
  for (int y = 0; y < yloops; y++)
  {
    int ypos = MINMAX_H * item_ct1.get_group(1) + y;
    if (ptbits & (1 << y) && pos < MEMWID)
    {
      points[2 * pos + 0] = xpos - 1;
      points[2 * pos + 1] = ypos;
      pos++;
    }
  }

  totbits = Shuffle(totbits, 31, item_ct1);
  if (tx < totbits)
  {
    int xpos = points[2 * tx + 0];
    int ypos = points[2 * tx + 1];
    int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f * val - data1[-1] - data1[1];
    float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
    float tra = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;
    if (tra * tra < edgeLimit * det)
    {
      float edge = (tra * tra) / det;
      float dx = 0.5f * (data1[1] - data1[-1]);
      float dy = 0.5f * (data1[pitch] - data1[-pitch]);
      float *data0 = d_Data0 + ptr - height * pitch;
      float *data2 = d_Data0 + ptr + height * pitch;
      float ds = 0.5f * (data0[0] - data2[0]);
      float dss = 2.0f * val - data2[0] - data0[0];
      float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy * dss - dys * dys;
      float idxy = dys * dxs - dxy * dss;
      float idxs = dxy * dys - dyy * dxs;
      float idet = 1.0f / (idxx * dxx + idxy * dxy + idxs * dxs);
      float idyy = dxx * dss - dxs * dxs;
      float idys = dxy * dxs - dxx * dys;
      float idss = dxx * dyy - dxy * dxy;
      float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
      float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
      float pds = idet * (idxs * dx + idys * dy + idss * ds);
      if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f)
      {
        pdx = dx / dxx;
        pdy = dy / dyy;
        pds = ds / dss;
      }
      float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
      int maxPts = d_MaxNumPoints;
      float sc =
          dpct::pow(2.0f, (float)scale / NUM_SCALES) * sycl::exp2(pds * factor);
      if (sc >= lowestScale)
      {
        dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
            &d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
        unsigned int idx = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            &d_PointCounter[2 * octave + 0], 0x7fffffff);
        idx = (idx >= maxPts ? maxPts - 1 : idx);
        d_Sift[idx].xpos = xpos + pdx;
        d_Sift[idx].ypos = ypos + pdy;
        d_Sift[idx].scale = sc;
        d_Sift[idx].sharpness = val + dval;
        d_Sift[idx].edgeness = edge;
        d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}

// __global__ void FindPointsMulti(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
// {
// #define MEMWID (MINMAX_W + 2)
//   __shared__ unsigned int cnt;
//   __shared__ unsigned short points[3 * MEMWID];


//   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
//   {
//     atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
//     atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
//   }
//   int tx = threadIdx.x;
//   int block = blockIdx.x / NUM_SCALES;
//   int scale = blockIdx.x - NUM_SCALES * block;
//   int minx = block * MINMAX_W;
//   int maxx = min(minx + MINMAX_W, width);
//   int xpos = minx + tx;
//   int size = pitch * height;
//   int ptr = size * scale + max(min(xpos - 1, width - 1), 0);

//   int yloops = min(height - MINMAX_H * blockIdx.y, MINMAX_H);
//   float maxv = 0.0f;
//   for (int y = 0; y < yloops; y++)
//   {
//     int ypos = MINMAX_H * blockIdx.y + y;
//     int yptr1 = ptr + ypos * pitch;
//     float val = d_Data0[yptr1 + 1 * size];
//     maxv = fmaxf(maxv, fabs(val));
//   }
//   maxv = fmaxf(maxv, ShiftDown(maxv, 16, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 8, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 4, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 2, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 1, MINMAX_W));
//   if (Shuffle(maxv, 0) <= thresh)
//     return;

//   if (tx == 0)
//     cnt = 0;
//   __syncthreads();

//   for (int y = 0; y < yloops; y++)
//   {

//     int ypos = MINMAX_H * blockIdx.y + y;
//     int yptr1 = ptr + ypos * pitch;
//     int yptr0 = ptr + max(0, ypos - 1) * pitch;
//     int yptr2 = ptr + min(height - 1, ypos + 1) * pitch;
//     float d20 = d_Data0[yptr0 + 1 * size];
//     float d21 = d_Data0[yptr1 + 1 * size];
//     float d22 = d_Data0[yptr2 + 1 * size];
//     float d31 = d_Data0[yptr1 + 2 * size];
//     float d11 = d_Data0[yptr1];

//     float d10 = d_Data0[yptr0];
//     float d12 = d_Data0[yptr2];
//     float ymin1 = fminf(fminf(d10, d11), d12);
//     float ymax1 = fmaxf(fmaxf(d10, d11), d12);
//     float d30 = d_Data0[yptr0 + 2 * size];
//     float d32 = d_Data0[yptr2 + 2 * size];
//     float ymin3 = fminf(fminf(d30, d31), d32);
//     float ymax3 = fmaxf(fmaxf(d30, d31), d32);
//     float ymin2 = fminf(fminf(ymin1, fminf(fminf(d20, d22), d21)), ymin3);
//     float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d20, d22), d21)), ymax3);

//     float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
//     float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
//     if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
//     {
//       if (d21 < -thresh)
//       {
//         float minv = fminf(fminf(nmin2, ymin1), ymin3);
//         minv = fminf(fminf(minv, d20), d22);
//         if (d21 < minv)
//         {
//           int pos = atomicInc(&cnt, MEMWID - 1);
//           points[3 * pos + 0] = xpos - 1;
//           points[3 * pos + 1] = ypos;
//           points[3 * pos + 2] = scale;
//         }
//       }
//       if (d21 > thresh)
//       {
//         float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
//         maxv = fmaxf(fmaxf(maxv, d20), d22);
//         if (d21 > maxv)
//         {
//           int pos = atomicInc(&cnt, MEMWID - 1);
//           points[3 * pos + 0] = xpos - 1;
//           points[3 * pos + 1] = ypos;
//           points[3 * pos + 2] = scale;
//         }
//       }
//     }
//   }
//   if (tx < cnt)
//   {
//     int xpos = points[3 * tx + 0];
//     int ypos = points[3 * tx + 1];
//     int scale = points[3 * tx + 2];
//     int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
//     float val = d_Data0[ptr];
//     float *data1 = &d_Data0[ptr];
//     float dxx = 2.0f * val - data1[-1] - data1[1];
//     float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
//     float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
//     float tra = dxx + dyy;
//     float det = dxx * dyy - dxy * dxy;
//     if (tra * tra < edgeLimit * det)
//     {
//       float edge = __fdividef(tra * tra, det);
//       float dx = 0.5f * (data1[1] - data1[-1]);
//       float dy = 0.5f * (data1[pitch] - data1[-pitch]);
//       float *data0 = d_Data0 + ptr - height * pitch;
//       float *data2 = d_Data0 + ptr + height * pitch;
//       float ds = 0.5f * (data0[0] - data2[0]);
//       float dss = 2.0f * val - data2[0] - data0[0];
//       float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
//       float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
//       float idxx = dyy * dss - dys * dys;
//       float idxy = dys * dxs - dxy * dss;
//       float idxs = dxy * dys - dyy * dxs;
//       float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
//       float idyy = dxx * dss - dxs * dxs;
//       float idys = dxy * dxs - dxx * dys;
//       float idss = dxx * dyy - dxy * dxy;
//       float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
//       float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
//       float pds = idet * (idxs * dx + idys * dy + idss * ds);
//       if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f)
//       {
//         pdx = __fdividef(dx, dxx);
//         pdy = __fdividef(dy, dyy);
//         pds = __fdividef(ds, dss);
//       }
//       float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
//       int maxPts = d_MaxNumPoints;
//       float sc = powf(2.0f, (float)scale / NUM_SCALES) * exp2f(pds * factor);
//       if (sc >= lowestScale)
//       {
//         atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
//         unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 0], 0x7fffffff);
//         idx = (idx >= maxPts ? maxPts - 1 : idx);
//         d_Sift[idx].xpos = xpos + pdx;
//         d_Sift[idx].ypos = ypos + pdy;
//         d_Sift[idx].scale = sc;
//         d_Sift[idx].sharpness = val + dval;
//         d_Sift[idx].edgeness = edge;
//         d_Sift[idx].subsampling = subsampling;
//       }
//     }
//   }
// }

// __global__ void FindPointsMultiOld(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
// {
// #define MEMWID (MINMAX_W + 2)
//   __shared__ float ymin1[MEMWID], ymin2[MEMWID], ymin3[MEMWID];
//   __shared__ float ymax1[MEMWID], ymax2[MEMWID], ymax3[MEMWID];
//   __shared__ unsigned int cnt;
//   __shared__ unsigned short points[3 * MEMWID];

//   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
//   {
//     atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
//     atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
//   }
//   int tx = threadIdx.x;
//   int block = blockIdx.x / NUM_SCALES;
//   int scale = blockIdx.x - NUM_SCALES * block;
//   int minx = block * MINMAX_W;
//   int maxx = min(minx + MINMAX_W, width);
//   int xpos = minx + tx;
//   int size = pitch * height;
//   int ptr = size * scale + max(min(xpos - 1, width - 1), 0);

//   int yloops = min(height - MINMAX_H * blockIdx.y, MINMAX_H);
//   float maxv = 0.0f;
//   for (int y = 0; y < yloops; y++)
//   {
//     int ypos = MINMAX_H * blockIdx.y + y;
//     int yptr1 = ptr + ypos * pitch;
//     float val = d_Data0[yptr1 + 1 * size];
//     maxv = fmaxf(maxv, fabs(val));
//   }
//   maxv = fmaxf(maxv, ShiftDown(maxv, 16, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 8, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 4, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 2, MINMAX_W));
//   maxv = fmaxf(maxv, ShiftDown(maxv, 1, MINMAX_W));
//   if (Shuffle(maxv, 0) <= thresh)
//     return;

//   if (tx == 0)
//     cnt = 0;
//   __syncthreads();

//   for (int y = 0; y < yloops; y++)
//   {

//     int ypos = MINMAX_H * blockIdx.y + y;
//     int yptr1 = ptr + ypos * pitch;
//     int yptr0 = ptr + max(0, ypos - 1) * pitch;
//     int yptr2 = ptr + min(height - 1, ypos + 1) * pitch;
//     float d20 = d_Data0[yptr0 + 1 * size];
//     float d21 = d_Data0[yptr1 + 1 * size];
//     float d22 = d_Data0[yptr2 + 1 * size];
//     float d31 = d_Data0[yptr1 + 2 * size];
//     float d11 = d_Data0[yptr1];

//     float d10 = d_Data0[yptr0];
//     float d12 = d_Data0[yptr2];
//     ymin1[tx] = fminf(fminf(d10, d11), d12);
//     ymax1[tx] = fmaxf(fmaxf(d10, d11), d12);
//     float d30 = d_Data0[yptr0 + 2 * size];
//     float d32 = d_Data0[yptr2 + 2 * size];
//     ymin3[tx] = fminf(fminf(d30, d31), d32);
//     ymax3[tx] = fmaxf(fmaxf(d30, d31), d32);
//     ymin2[tx] = fminf(fminf(ymin1[tx], fminf(fminf(d20, d22), d21)), ymin3[tx]);
//     ymax2[tx] = fmaxf(fmaxf(ymax1[tx], fmaxf(fmaxf(d20, d22), d21)), ymax3[tx]);

//     __syncthreads();

//     if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
//     {
//       if (d21 < -thresh)
//       {
//         float minv = fminf(fminf(fminf(ymin2[tx - 1], ymin2[tx + 1]), ymin1[tx]), ymin3[tx]);
//         minv = fminf(fminf(minv, d20), d22);
//         if (d21 < minv)
//         {
//           int pos = atomicInc(&cnt, MEMWID - 1);
//           points[3 * pos + 0] = xpos - 1;
//           points[3 * pos + 1] = ypos;
//           points[3 * pos + 2] = scale;
//         }
//       }
//       if (d21 > thresh)
//       {
//         float maxv = fmaxf(fmaxf(fmaxf(ymax2[tx - 1], ymax2[tx + 1]), ymax1[tx]), ymax3[tx]);
//         maxv = fmaxf(fmaxf(maxv, d20), d22);
//         if (d21 > maxv)
//         {
//           int pos = atomicInc(&cnt, MEMWID - 1);
//           points[3 * pos + 0] = xpos - 1;
//           points[3 * pos + 1] = ypos;
//           points[3 * pos + 2] = scale;
//         }
//       }
//     }
//     __syncthreads();
//   }
//   if (tx < cnt)
//   {
//     int xpos = points[3 * tx + 0];
//     int ypos = points[3 * tx + 1];
//     int scale = points[3 * tx + 2];
//     int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
//     float val = d_Data0[ptr];
//     float *data1 = &d_Data0[ptr];
//     float dxx = 2.0f * val - data1[-1] - data1[1];
//     float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
//     float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
//     float tra = dxx + dyy;
//     float det = dxx * dyy - dxy * dxy;
//     if (tra * tra < edgeLimit * det)
//     {
//       float edge = __fdividef(tra * tra, det);
//       float dx = 0.5f * (data1[1] - data1[-1]);
//       float dy = 0.5f * (data1[pitch] - data1[-pitch]);
//       float *data0 = d_Data0 + ptr - height * pitch;
//       float *data2 = d_Data0 + ptr + height * pitch;
//       float ds = 0.5f * (data0[0] - data2[0]);
//       float dss = 2.0f * val - data2[0] - data0[0];
//       float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
//       float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
//       float idxx = dyy * dss - dys * dys;
//       float idxy = dys * dxs - dxy * dss;
//       float idxs = dxy * dys - dyy * dxs;
//       float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
//       float idyy = dxx * dss - dxs * dxs;
//       float idys = dxy * dxs - dxx * dys;
//       float idss = dxx * dyy - dxy * dxy;
//       float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
//       float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
//       float pds = idet * (idxs * dx + idys * dy + idss * ds);
//       if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f)
//       {
//         pdx = __fdividef(dx, dxx);
//         pdy = __fdividef(dy, dyy);
//         pds = __fdividef(ds, dss);
//       }
//       float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
//       int maxPts = d_MaxNumPoints;
//       float sc = powf(2.0f, (float)scale / NUM_SCALES) * exp2f(pds * factor);
//       if (sc >= lowestScale)
//       {
//         unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 0], 0x7fffffff);
//         idx = (idx >= maxPts ? maxPts - 1 : idx);
//         d_Sift[idx].xpos = xpos + pdx;
//         d_Sift[idx].ypos = ypos + pdy;
//         d_Sift[idx].scale = sc;
//         d_Sift[idx].sharpness = val + dval;
//         d_Sift[idx].edgeness = edge;
//         d_Sift[idx].subsampling = subsampling;
//       }
//     }
//   }
// }

void LaplaceMultiTex(dpct::image_accessor_ext<float, 2> texObj, float *d_Result,
                     int width, int pitch, int height, int octave,
                     const sycl::nd_item<3> &item_ct1,
                     float const *d_LaplaceKernel, float *data1, float *data2)
{

  const int tx = item_ct1.get_local_id(2);
  const int xp = item_ct1.get_group(2) * LAPLACE_W + tx;
  const int yp = item_ct1.get_group(1);
  const int scale = item_ct1.get_local_id(1);
  float *kernel =
      const_cast<float *>(d_LaplaceKernel + octave * 12 * 16 + scale * 16);
  float *sdata1 = data1 + (LAPLACE_W + 2 * LAPLACE_R) * scale;
  float x = xp - 3.5;
  float y = yp + 0.5;
  sdata1[tx] = kernel[0] * texObj.read(x, y) +
               kernel[1] * (texObj.read(x, y - 1.0) + texObj.read(x, y + 1.0)) +
               kernel[2] * (texObj.read(x, y - 2.0) + texObj.read(x, y + 2.0)) +
               kernel[3] * (texObj.read(x, y - 3.0) + texObj.read(x, y + 3.0)) +
               kernel[4] * (texObj.read(x, y - 4.0) + texObj.read(x, y + 4.0));
  item_ct1.barrier(sycl::access::fence_space::local_space);
  float *sdata2 = data2 + LAPLACE_W * scale;
  if (tx < LAPLACE_W)
  {
    sdata2[tx] = kernel[0] * sdata1[tx + 4] +
                 kernel[1] * (sdata1[tx + 3] + sdata1[tx + 5]) +
                 kernel[2] * (sdata1[tx + 2] + sdata1[tx + 6]) +
                 kernel[3] * (sdata1[tx + 1] + sdata1[tx + 7]) +
                 kernel[4] * (sdata1[tx + 0] + sdata1[tx + 8]);
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if (tx < LAPLACE_W && scale < LAPLACE_S - 1 && xp < width)
    d_Result[scale * height * pitch + yp * pitch + xp] = sdata2[tx] - sdata2[tx + LAPLACE_W];
}

/*
DPCT1110:46: The total declared local variable size in device function
LaplaceMultiMem exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void LaplaceMultiMem(float *d_Image, float *d_Result, int width, int pitch,
                     int height, int octave, const sycl::nd_item<3> &item_ct1,
                     float const *d_LaplaceKernel, float *buff)
{

  const int tx = item_ct1.get_local_id(2);
  const int xp = item_ct1.get_group(2) * LAPLACE_W + tx;
  const int yp = item_ct1.get_group(1);
  float *data = d_Image + sycl::max(sycl::min(xp - LAPLACE_R, width - 1),
                                    0); // multiply with 4 for max func
  float temp[2 * LAPLACE_R + 1];

  float kern[LAPLACE_S][LAPLACE_R + 1];
  if (xp < (width + 2 * LAPLACE_R))
  {
    for (int i = 0; i <= 2 * LAPLACE_R; i++)
      temp[i] =
          data[sycl::max(0, sycl::min(yp + i - LAPLACE_R, height - 1)) * pitch];
    for (int scale = 0; scale < LAPLACE_S; scale++)
    {
      float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
      float *kernel =
          const_cast<float *>(d_LaplaceKernel + octave * 12 * 16 + scale * 16);
      for (int i = 0; i <= LAPLACE_R; i++)
      {
        kern[scale][i] = kernel[i];
      }
      float sum = kern[scale][0] * temp[LAPLACE_R];
#pragma unroll
      for (int j = 1; j <= LAPLACE_R; j++)
        sum += kern[scale][j] * (temp[LAPLACE_R - j] + temp[LAPLACE_R + j]);
      buf[tx] = sum;
    }
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if (tx < LAPLACE_W && xp < (width + 2 * LAPLACE_R))
  {
    int scale = 0;
    float oldRes = kern[scale][0] * buff[tx + LAPLACE_R];

#pragma unroll
    for (int j = 1; j <= LAPLACE_R; j++)
      oldRes += kern[scale][j] * (buff[tx + LAPLACE_R - j] + buff[tx + LAPLACE_R + j]);

    for (int scale = 1; scale < LAPLACE_S; scale++)
    {
      float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;

      float res = kern[scale][0] * buf[tx + LAPLACE_R];

#pragma unroll
      for (int j = 1; j <= LAPLACE_R; j++)
        res += kern[scale][j] * (buf[tx + LAPLACE_R - j] + buf[tx + LAPLACE_R + j]);

      d_Result[(scale - 1) * height * pitch + yp * pitch + xp] = res - oldRes;
      oldRes = res;
    }
  }
}

// __global__ void LaplaceMultiMemWide(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
// {
//   __shared__ float buff[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
//   const int tx = threadIdx.x;
//   const int xp = blockIdx.x * LAPLACE_W + tx;
//   const int xp4 = blockIdx.x * LAPLACE_W + 4 * tx;
//   const int yp = blockIdx.y;
//   float kern[LAPLACE_S][LAPLACE_R + 1];
//   float *data = d_Image + max(min(xp - 4, width - 1), 0);
//   float temp[9];
//   if (xp < (width + 2 * LAPLACE_R))
//   {
//     for (int i = 0; i < 4; i++)
//       temp[i] = data[max(0, min(yp + i - 4, height - 1)) * pitch];
//     for (int i = 4; i < 8 + 1; i++)
//       temp[i] = data[min(yp + i - 4, height - 1) * pitch];
//     for (int scale = 0; scale < LAPLACE_S; scale++)
//     {
//       float *kernel = d_LaplaceKernel + octave * 12 * 16 + scale * 16;
//       for (int i = 0; i <= LAPLACE_R; i++)
//         kern[scale][i] = kernel[LAPLACE_R - i];
//       float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
//       buf[tx] = kern[scale][4] * temp[4] +
//                 kern[scale][3] * (temp[3] + temp[5]) + kern[scale][2] * (temp[2] + temp[6]) +
//                 kern[scale][1] * (temp[1] + temp[7]) + kern[scale][0] * (temp[0] + temp[8]);
//     }
//   }
//   __syncthreads();
//   if (tx < LAPLACE_W / 4 && xp4 < width)
//   {
//     float4 b0 = reinterpret_cast<float4 *>(buff)[tx + 0];
//     float4 b1 = reinterpret_cast<float4 *>(buff)[tx + 1];
//     float4 b2 = reinterpret_cast<float4 *>(buff)[tx + 2];
//     float4 old4, new4, dif4;
//     old4.x = kern[0][4] * b1.x + kern[0][3] * (b0.w + b1.y) + kern[0][2] * (b0.z + b1.z) +
//              kern[0][1] * (b0.y + b1.w) + kern[0][0] * (b0.x + b2.x);
//     old4.y = kern[0][4] * b1.y + kern[0][3] * (b1.x + b1.z) + kern[0][2] * (b0.w + b1.w) +
//              kern[0][1] * (b0.z + b2.x) + kern[0][0] * (b0.y + b2.y);
//     old4.z = kern[0][4] * b1.z + kern[0][3] * (b1.y + b1.w) + kern[0][2] * (b1.x + b2.x) +
//              kern[0][1] * (b0.w + b2.y) + kern[0][0] * (b0.z + b2.z);
//     old4.w = kern[0][4] * b1.w + kern[0][3] * (b1.z + b2.x) + kern[0][2] * (b1.y + b2.y) +
//              kern[0][1] * (b1.x + b2.z) + kern[0][0] * (b0.w + b2.w);
//     for (int scale = 1; scale < LAPLACE_S; scale++)
//     {
//       float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
//       float4 b0 = reinterpret_cast<float4 *>(buf)[tx + 0];
//       float4 b1 = reinterpret_cast<float4 *>(buf)[tx + 1];
//       float4 b2 = reinterpret_cast<float4 *>(buf)[tx + 2];
//       new4.x = kern[scale][4] * b1.x + kern[scale][3] * (b0.w + b1.y) +
//                kern[scale][2] * (b0.z + b1.z) + kern[scale][1] * (b0.y + b1.w) +
//                kern[scale][0] * (b0.x + b2.x);
//       new4.y = kern[scale][4] * b1.y + kern[scale][3] * (b1.x + b1.z) +
//                kern[scale][2] * (b0.w + b1.w) + kern[scale][1] * (b0.z + b2.x) +
//                kern[scale][0] * (b0.y + b2.y);
//       new4.z = kern[scale][4] * b1.z + kern[scale][3] * (b1.y + b1.w) +
//                kern[scale][2] * (b1.x + b2.x) + kern[scale][1] * (b0.w + b2.y) +
//                kern[scale][0] * (b0.z + b2.z);
//       new4.w = kern[scale][4] * b1.w + kern[scale][3] * (b1.z + b2.x) +
//                kern[scale][2] * (b1.y + b2.y) + kern[scale][1] * (b1.x + b2.z) +
//                kern[scale][0] * (b0.w + b2.w);
//       dif4.x = new4.x - old4.x;
//       dif4.y = new4.y - old4.y;
//       dif4.z = new4.z - old4.z;
//       dif4.w = new4.w - old4.w;
//       reinterpret_cast<float4 *>(&d_Result[(scale - 1) * height * pitch + yp * pitch + xp4])[0] = dif4;
//       old4 = new4;
//     }
//   }
// }

// __global__ void LaplaceMultiMemTest(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
// {
//   __shared__ float data1[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
//   __shared__ float data2[LAPLACE_W * LAPLACE_S];
//   const int tx = threadIdx.x;
//   const int xp = blockIdx.x * LAPLACE_W + tx;
//   const int yp = LAPLACE_H * blockIdx.y;
//   const int scale = threadIdx.y;
//   float *kernel = d_LaplaceKernel + octave * 12 * 16 + scale * 16;
//   float *sdata1 = data1 + (LAPLACE_W + 2 * LAPLACE_R) * scale;
//   float *data = d_Image + max(min(xp - 4, width - 1), 0);
//   int h = height - 1;
//   float temp[8 + LAPLACE_H], kern[LAPLACE_R + 1];
//   for (int i = 0; i < 4; i++)
//     temp[i] = data[max(0, min(yp + i - 4, h)) * pitch];
//   for (int i = 4; i < 8 + LAPLACE_H; i++)
//     temp[i] = data[min(yp + i - 4, h) * pitch];
//   for (int i = 0; i <= LAPLACE_R; i++)
//     kern[i] = kernel[LAPLACE_R - i];
//   for (int j = 0; j < LAPLACE_H; j++)
//   {
//     sdata1[tx] = kern[4] * temp[4 + j] +
//                  kern[3] * (temp[3 + j] + temp[5 + j]) + kern[2] * (temp[2 + j] + temp[6 + j]) +
//                  kern[1] * (temp[1 + j] + temp[7 + j]) + kern[0] * (temp[0 + j] + temp[8 + j]);
//     __syncthreads();
//     float *sdata2 = data2 + LAPLACE_W * scale;
//     if (tx < LAPLACE_W)
//     {
//       sdata2[tx] = kern[4] * sdata1[tx + 4] +
//                    kern[3] * (sdata1[tx + 3] + sdata1[tx + 5]) + kern[2] * (sdata1[tx + 2] + sdata1[tx + 6]) +
//                    kern[1] * (sdata1[tx + 1] + sdata1[tx + 7]) + kern[0] * (sdata1[tx + 0] + sdata1[tx + 8]);
//     }
//     __syncthreads();
//     if (tx < LAPLACE_W && scale < LAPLACE_S - 1 && xp < width && (yp + j) < height)
//       d_Result[scale * height * pitch + (yp + j) * pitch + xp] = sdata2[tx] - sdata2[tx + LAPLACE_W];
//   }
// }

// __global__ void LaplaceMultiMemOld(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
// {
//   __shared__ float data1[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
//   __shared__ float data2[LAPLACE_W * LAPLACE_S];
//   const int tx = threadIdx.x;
//   const int xp = blockIdx.x * LAPLACE_W + tx;
//   const int yp = blockIdx.y;
//   const int scale = threadIdx.y;
//   float *kernel = d_LaplaceKernel + octave * 12 * 16 + scale * 16;
//   float *sdata1 = data1 + (LAPLACE_W + 2 * LAPLACE_R) * scale;
//   float *data = d_Image + max(min(xp - 4, width - 1), 0);
//   int h = height - 1;
//   sdata1[tx] = kernel[0] * data[min(yp, h) * pitch] +
//                kernel[1] * (data[max(0, min(yp - 1, h)) * pitch] + data[min(yp + 1, h) * pitch]) +
//                kernel[2] * (data[max(0, min(yp - 2, h)) * pitch] + data[min(yp + 2, h) * pitch]) +
//                kernel[3] * (data[max(0, min(yp - 3, h)) * pitch] + data[min(yp + 3, h) * pitch]) +
//                kernel[4] * (data[max(0, min(yp - 4, h)) * pitch] + data[min(yp + 4, h) * pitch]);
//   __syncthreads();
//   float *sdata2 = data2 + LAPLACE_W * scale;
//   if (tx < LAPLACE_W)
//   {
//     sdata2[tx] = kernel[0] * sdata1[tx + 4] +
//                  kernel[1] * (sdata1[tx + 3] + sdata1[tx + 5]) +
//                  kernel[2] * (sdata1[tx + 2] + sdata1[tx + 6]) +
//                  kernel[3] * (sdata1[tx + 1] + sdata1[tx + 7]) +
//                  kernel[4] * (sdata1[tx + 0] + sdata1[tx + 8]);
//   }
//   __syncthreads();
//   if (tx < LAPLACE_W && scale < LAPLACE_S - 1 && xp < width)
//     d_Result[scale * height * pitch + yp * pitch + xp] = sdata2[tx] - sdata2[tx + LAPLACE_W];
// }

void LowPass(float *d_Image, float *d_Result, int width, int pitch, int height,
             const sycl::nd_item<3> &item_ct1, float const *d_LowPassKernel,
             float *buffer)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int xp = item_ct1.get_group(2) * LOWPASS_W + tx;
  const int yp = item_ct1.get_group(1) * LOWPASS_H + ty;
  float *kernel = const_cast<float *>(d_LowPassKernel);
  float *data = d_Image + sycl::max(sycl::min(xp - 4, width - 1), 0);
  float *buff = buffer + ty * (LOWPASS_W + 2 * LOWPASS_R);
  int h = height - 1;
  if (yp < height)
    buff[tx] = kernel[4] * data[sycl::min(yp, h) * pitch] +
               kernel[3] * (data[sycl::max(0, sycl::min(yp - 1, h)) * pitch] +
                            data[sycl::min(yp + 1, h) * pitch]) +
               kernel[2] * (data[sycl::max(0, sycl::min(yp - 2, h)) * pitch] +
                            data[sycl::min(yp + 2, h) * pitch]) +
               kernel[1] * (data[sycl::max(0, sycl::min(yp - 3, h)) * pitch] +
                            data[sycl::min(yp + 3, h) * pitch]) +
               kernel[0] * (data[sycl::max(0, sycl::min(yp - 4, h)) * pitch] +
                            data[sycl::min(yp + 4, h) * pitch]);
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if (tx < LOWPASS_W && xp < width && yp < height)
    d_Result[yp * pitch + xp] = kernel[4] * buff[tx + 4] +
                                kernel[3] * (buff[tx + 3] + buff[tx + 5]) + kernel[2] * (buff[tx + 2] + buff[tx + 6]) +
                                kernel[1] * (buff[tx + 1] + buff[tx + 7]) + kernel[0] * (buff[tx + 0] + buff[tx + 8]);
}

void LowPassBlockOld(float *d_Image, float *d_Result, int width, int pitch, int height,
                     const sycl::nd_item<3> &item_ct1,
                     float const *d_LowPassKernel,
                     sycl::local_accessor<float, 2> xrows)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int xp = item_ct1.get_group(2) * LOWPASS_W + tx;
  const int yp = item_ct1.get_group(1) * LOWPASS_H + ty;
  const int N = 16;
  float *k = const_cast<float *>(d_LowPassKernel);
  int xl = sycl::max(sycl::min(xp - 4, width - 1), 0);
  for (int l = -8; l <= LOWPASS_H; l += 4)
  {
    if (l < LOWPASS_H)
    {
      int yl = sycl::max(sycl::min(yp + l + 4, height - 1), 0);
      float val = d_Image[yl * pitch + xl];
      xrows[(l + 8 + ty) % N][tx] =
          k[4] * ShiftDown(val, 4, item_ct1) +
          k[3] * (ShiftDown(val, 5, item_ct1) + ShiftDown(val, 3, item_ct1)) +
          k[2] * (ShiftDown(val, 6, item_ct1) + ShiftDown(val, 2, item_ct1)) +
          k[1] * (ShiftDown(val, 7, item_ct1) + ShiftDown(val, 1, item_ct1)) +
          k[0] * (ShiftDown(val, 8, item_ct1) + val);
    }
    if (l >= 4)
    {
      int ys = yp + l - 4;
      if (xp < width && ys < height && tx < LOWPASS_W)
        d_Result[ys * pitch + xp] = k[4] * xrows[(l + 0 + ty) % N][tx] +
                                    k[3] * (xrows[(l - 1 + ty) % N][tx] + xrows[(l + 1 + ty) % N][tx]) +
                                    k[2] * (xrows[(l - 2 + ty) % N][tx] + xrows[(l + 2 + ty) % N][tx]) +
                                    k[1] * (xrows[(l - 3 + ty) % N][tx] + xrows[(l + 3 + ty) % N][tx]) +
                                    k[0] * (xrows[(l - 4 + ty) % N][tx] + xrows[(l + 4 + ty) % N][tx]);
    }
    if (l >= 0)
      /*
      DPCT1118:47: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
  }
}

void LowPassBlock(float *d_Image, float *d_Result, int width, int pitch, int height,
                  const sycl::nd_item<3> &item_ct1, float const *d_LowPassKernel,
                  sycl::local_accessor<float, 2> xrows)
{

  const int tx = item_ct1.get_local_id(2);
  const int ty = item_ct1.get_local_id(1);
  const int xp = item_ct1.get_group(2) * LOWPASS_W + tx;
  const int yp = item_ct1.get_group(1) * LOWPASS_H + ty;
  const int N = 16;
  float *k = const_cast<float *>(d_LowPassKernel);
  int xl = sycl::max(sycl::min(xp - 4, width - 1), 0);
#pragma unroll
  for (int l = -8; l < 4; l += 4)
  {
    int ly = l + ty;
    int yl = sycl::max(sycl::min(yp + l + 4, height - 1), 0);
    float val = d_Image[yl * pitch + xl]; // d_Image[yl*pitch + xl].x
    val = k[4] * ShiftDown(val, 4, item_ct1) +
          k[3] * (ShiftDown(val, 5, item_ct1) + ShiftDown(val, 3, item_ct1)) +
          k[2] * (ShiftDown(val, 6, item_ct1) + ShiftDown(val, 2, item_ct1)) +
          k[1] * (ShiftDown(val, 7, item_ct1) + ShiftDown(val, 1, item_ct1)) +
          k[0] * (ShiftDown(val, 8, item_ct1) + val);
    xrows[ly + 8][tx] = val;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
  for (int l = 4; l < LOWPASS_H; l += 4)
  {
    int ly = l + ty;
    int yl = sycl::min(yp + l + 4, height - 1);
    float val = d_Image[yl * pitch + xl];
    val = k[4] * ShiftDown(val, 4, item_ct1) +
          k[3] * (ShiftDown(val, 5, item_ct1) + ShiftDown(val, 3, item_ct1)) +
          k[2] * (ShiftDown(val, 6, item_ct1) + ShiftDown(val, 2, item_ct1)) +
          k[1] * (ShiftDown(val, 7, item_ct1) + ShiftDown(val, 1, item_ct1)) +
          k[0] * (ShiftDown(val, 8, item_ct1) + val);
    xrows[(ly + 8) % N][tx] = val;
    int ys = yp + l - 4;
    if (xp < width && ys < height && tx < LOWPASS_W)
      d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                  k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                  k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                  k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                  k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
    /*
    DPCT1118:48: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp < width && ys < height && tx < LOWPASS_W)
    d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
}
