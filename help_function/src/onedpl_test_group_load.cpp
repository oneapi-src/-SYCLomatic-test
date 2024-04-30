// ====------ onedpl_test_group_load.cpp------------ *- C++ -* ----===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <iostream>
#include <oneapi/dpl/iterator>

template<dpct::group::load_algorithm T>
bool test_load_blocked_striped() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT & dpct::group::load_algorithm::BLOCK_LOAD_STRIPED 
  // in its entirety as API functions
  sycl::queue q;
  oneapi::dpl::counting_iterator<int> count_it(0);
  sycl::buffer<int, 1> buffer(count_it, count_it + 512);
  
  q.submit([&](sycl::handler &h) {
    using group_load = dpct::group::workgroup_load<4, T, int,  int *, sycl::nd_item<3>>;
    size_t temp_storage_size = group_load::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), h);
    sycl::accessor data_accessor(buffer, h, sycl::read_write);
    
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = data_accessor.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          group_load(tmp).load(item, d, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_global_linear_id() * 4; // Each thread_data has 4 elements
            for (int i = 0; i < 4; ++i) {
                data_accessor[global_index + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();
  
  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  
  if ( T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT)
  {
   for (int i = 0; i < 512; ++i) {
    if (ptr[i] != i) {
      std::cout <<" failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(ptr, ptr + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout <<" pass\n";
  return true;
  }
  else{
  
   int expected[512]={0, 128, 256, 384, 1, 129, 257, 385, 2, 130, 258, 386, 3, 131, 259, 387, 4, 132, 260, 388, 5, 133, 261, 389, 6, 134, 262, 390, 7, 135, 263, 391, 8, 136, 264, 392, 9, 137, 265, 393, 10, 138, 266, 394, 11, 139, 267, 395, 12, 140, 268, 396, 13, 141, 269, 397, 14, 142, 270, 398, 15, 143, 271, 399, 4, 144, 272, 400, 132, 145, 273, 401, 260, 146, 274, 402, 388, 147, 275, 403, 5, 148, 276, 404, 133, 149, 277, 405, 261, 150, 278, 406, 389, 151, 279, 407, 6, 152, 280, 408, 134, 153, 281, 409, 262, 154, 282, 410, 390, 155, 283, 411, 7, 156, 284, 412, 135, 157, 285, 413, 263, 158, 286, 414, 391, 159, 287, 415, 8, 160, 288, 416, 136, 161, 289, 417, 264, 162, 290, 418, 392, 163, 291, 419, 9, 164, 292, 420, 137, 165, 293, 421, 265, 166, 294, 422, 393, 167, 295, 423, 10, 168, 296, 424, 138, 169, 297, 425, 266, 170, 298, 426, 394, 171, 299, 427, 11, 172, 300, 428, 139, 173, 301, 429, 267, 174, 302, 430, 395, 175, 303, 431, 12, 11, 304, 432, 140, 172, 305, 433, 268, 300, 306, 434, 396, 428, 307, 435, 13, 139, 308, 436, 141, 173, 309, 437, 269, 301, 310, 438, 397, 429, 311, 439, 14, 267, 312, 440, 142, 174, 313, 441, 270, 302, 314, 442, 398, 430, 315, 443, 15, 395, 316, 444, 143, 175, 317, 445, 271, 303, 318, 446, 399, 431, 319, 447, 4, 12, 320, 448, 144, 11, 321, 449, 272, 304, 322, 450, 400, 432, 323, 451, 132, 140, 324, 452, 145, 172, 325, 453, 273, 305, 326, 454, 401, 433, 327, 455, 260, 268, 328, 456, 146, 300, 329, 457, 274, 306, 330, 458, 402, 434, 331, 459, 388, 396, 332, 460, 147, 428, 333, 461, 275, 307, 334, 462, 403, 435, 335, 463, 5, 13, 336, 464, 148, 139, 337, 465, 276, 308, 338, 466, 404, 436, 339, 467, 133, 141, 340, 468, 149, 173, 341, 469, 277, 309, 342, 470, 405, 437, 343, 471, 261, 269, 344, 472, 150, 301, 345, 473, 278, 310, 346, 474, 406, 438, 347, 475, 389, 397, 348, 476, 151, 429, 349, 477, 279, 311, 350, 478, 407, 439, 351, 479, 6, 14, 261, 480, 152, 267, 269, 481, 280, 312, 344, 482, 408, 440, 472, 483, 134, 142, 150, 484, 153, 174, 301, 485, 281, 313, 345, 486, 409, 441, 473, 487, 262, 270, 278, 488, 154, 302, 310, 489, 282, 314, 346, 490, 410, 442, 474, 491, 390, 398, 406, 492, 155, 430, 438, 493, 283, 315, 347, 494, 411, 443, 475, 495, 7, 15, 389, 496, 156, 395, 397, 497, 284, 316, 348, 498, 412, 444, 476, 499, 135, 143, 151, 500, 157, 175, 429, 501, 285, 317, 349, 502, 413, 445, 477, 503, 263, 271, 279, 504, 158, 303, 311, 505, 286, 318, 350, 506, 414, 446, 478, 507, 391, 399, 407, 508, 159, 431, 439, 509, 287, 319, 351, 510, 415, 447, 479, 511};
  for (int i = 0; i < 512; ++i) {
      if (ptr[i] != expected[i]) {
        std::cout <<" failed\n";
        std::ostream_iterator<int> Iter(std::cout, ", ");
        std::copy(ptr, ptr + 512, Iter);
        std::cout << std::endl;
        return false;
      }
    }
  
    std::cout <<" pass\n";
    return true;
  }
}

bool test_load_subgroup_striped_standalone() {
  // Tests dpct::group::load_subgroup_striped as standalone method
  sycl::queue q;
  int data[512];
  for (int i = 0; i < 512; i++) data[i] = i;

  sycl::buffer<int, 1> buffer(data, 512);
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          dpct::group::uninitialized_load_subgroup_striped<4, int>(item, d, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_global_linear_id() * 4; // Each thread_data has 4 elements
          for (int i = 0; i < 4; ++i) {
                dacc[global_index + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  int expected[512] = {0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63, 64, 80, 96, 112, 65, 81, 97, 113, 66, 82, 98, 114, 67, 83, 99, 115, 68, 84, 100, 116, 69, 85, 101, 117, 70, 86, 102, 118, 71, 87, 103, 119, 72, 88, 104, 120, 73, 89, 105, 121, 74, 90, 106, 122, 75, 91, 107, 123, 76, 92, 108, 124, 77, 93, 109, 125, 78, 94, 110, 126, 79, 95, 111, 127, 128, 144, 160, 176, 129, 145, 161, 177, 130, 146, 162, 178, 131, 147, 163, 179, 132, 148, 164, 180, 133, 149, 165, 181, 134, 150, 166, 182, 135, 151, 167, 183, 136, 152, 168, 184, 137, 153, 169, 185, 138, 154, 170, 186, 139, 155, 171, 187, 140, 156, 172, 188, 141, 157, 173, 189, 142, 158, 174, 190, 143, 159, 175, 191, 192, 208, 224, 240, 193, 209, 225, 241, 194, 210, 226, 242, 195, 211, 227, 243, 196, 212, 228, 244, 197, 213, 229, 245, 198, 214, 230, 246, 199, 215, 231, 247, 200, 216, 232, 248, 201, 217, 233, 249, 202, 218, 234, 250, 203, 219, 235, 251, 204, 220, 236, 252, 205, 221, 237, 253, 206, 222, 238, 254, 207, 223, 239, 255, 256, 272, 288, 304, 257, 273, 289, 305, 258, 274, 290, 306, 259, 275, 291, 307, 260, 276, 292, 308, 261, 277, 293, 309, 262, 278, 294, 310, 263, 279, 295, 311, 264, 280, 296, 312, 265, 281, 297, 313, 266, 282, 298, 314, 267, 283, 299, 315, 268, 284, 300, 316, 269, 285, 301, 317, 270, 286, 302, 318, 271, 287, 303, 319, 320, 336, 352, 368, 321, 337, 353, 369, 322, 338, 354, 370, 323, 339, 355, 371, 324, 340, 356, 372, 325, 341, 357, 373, 326, 342, 358, 374, 327, 343, 359, 375, 328, 344, 360, 376, 329, 345, 361, 377, 330, 346, 362, 378, 331, 347, 363, 379, 332, 348, 364, 380, 333, 349, 365, 381, 334, 350, 366, 382, 335, 351, 367, 383, 384, 400, 416, 432, 385, 401, 417, 433, 386, 402, 418, 434, 387, 403, 419, 435, 388, 404, 420, 436, 389, 405, 421, 437, 390, 406, 422, 438, 391, 407, 423, 439, 392, 408, 424, 440, 393, 409, 425, 441, 394, 410, 426, 442, 395, 411, 427, 443, 396, 412, 428, 444, 397, 413, 429, 445, 398, 414, 430, 446, 399, 415, 431, 447, 448, 464, 480, 496, 449, 465, 481, 497, 450, 466, 482, 498, 451, 467, 483, 499, 452, 468, 484, 500, 453, 469, 485, 501, 454, 470, 486, 502, 455, 471, 487, 503, 456, 472, 488, 504, 457, 473, 489, 505, 458, 474, 490, 506, 459, 475, 491, 507, 460, 476, 492, 508, 461, 477, 493, 509, 462, 478, 494, 510, 463, 479, 495, 511};
  
  for (int i = 0; i < 512; ++i) {
    if (ptr[i] != expected[i]) {
      std::cout <<" failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(ptr, ptr + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout <<" pass\n";
  return true;
}

template<dpct::group::load_algorithm T>
bool test_load_blocked_striped_standalone() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT & dpct::group::load_algorithm::BLOCK_LOAD_STRIPED 
  // as standalone methods
  sycl::queue q;
  int data[512];
  for (int i = 0; i < 512; i++) data[i] = i;

  sycl::buffer<int, 1> buffer(data, 512);
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          if( T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT)
            {dpct::group::load_blocked<4, T, int>(item, d, thread_data);}
          else
            {dpct::group::load_striped<4, T, int>(item, d, thread_data);}
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_global_linear_id() * 4; // Each thread_data has 4 elements
            for (int i = 0; i < 4; ++i) {
                dacc[global_index + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  if(T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT)
   {
     for (int i = 0; i < 512; ++i) {
      if (ptr[i] != i) {
        std::cout <<" failed\n";
        std::ostream_iterator<int> Iter(std::cout, ", ");
        std::copy(ptr, ptr + 512, Iter);
        std::cout << std::endl;
        return false;
        }
      }
    
      std::cout <<" pass\n";
      return true;
  }
  else{
    int expected[512]={0, 128, 256, 384, 1, 129, 257, 385, 2, 130, 258, 386, 3, 131, 259, 387, 4, 132, 260, 388, 5, 133, 261, 389, 6, 134, 262, 390, 7, 135, 263, 391, 8, 136, 264, 392, 9, 137, 265, 393, 10, 138, 266, 394, 11, 139, 267, 395, 12, 140, 268, 396, 13, 141, 269, 397, 14, 142, 270, 398, 15, 143, 271, 399, 4, 144, 272, 400, 132, 145, 273, 401, 260, 146, 274, 402, 388, 147, 275, 403, 5, 148, 276, 404, 133, 149, 277, 405, 261, 150, 278, 406, 389, 151, 279, 407, 6, 152, 280, 408, 134, 153, 281, 409, 262, 154, 282, 410, 390, 155, 283, 411, 7, 156, 284, 412, 135, 157, 285, 413, 263, 158, 286, 414, 391, 159, 287, 415, 8, 160, 288, 416, 136, 161, 289, 417, 264, 162, 290, 418, 392, 163, 291, 419, 9, 164, 292, 420, 137, 165, 293, 421, 265, 166, 294, 422, 393, 167, 295, 423, 10, 168, 296, 424, 138, 169, 297, 425, 266, 170, 298, 426, 394, 171, 299, 427, 11, 172, 300, 428, 139, 173, 301, 429, 267, 174, 302, 430, 395, 175, 303, 431, 12, 11, 304, 432, 140, 172, 305, 433, 268, 300, 306, 434, 396, 428, 307, 435, 13, 139, 308, 436, 141, 173, 309, 437, 269, 301, 310, 438, 397, 429, 311, 439, 14, 267, 312, 440, 142, 174, 313, 441, 270, 302, 314, 442, 398, 430, 315, 443, 15, 395, 316, 444, 143, 175, 317, 445, 271, 303, 318, 446, 399, 431, 319, 447, 4, 12, 320, 448, 144, 11, 321, 449, 272, 304, 322, 450, 400, 432, 323, 451, 132, 140, 324, 452, 145, 172, 325, 453, 273, 305, 326, 454, 401, 433, 327, 455, 260, 268, 328, 456, 146, 300, 329, 457, 274, 306, 330, 458, 402, 434, 331, 459, 388, 396, 332, 460, 147, 428, 333, 461, 275, 307, 334, 462, 403, 435, 335, 463, 5, 13, 336, 464, 148, 139, 337, 465, 276, 308, 338, 466, 404, 436, 339, 467, 133, 141, 340, 468, 149, 173, 341, 469, 277, 309, 342, 470, 405, 437, 343, 471, 261, 269, 344, 472, 150, 301, 345, 473, 278, 310, 346, 474, 406, 438, 347, 475, 389, 397, 348, 476, 151, 429, 349, 477, 279, 311, 350, 478, 407, 439, 351, 479, 6, 14, 261, 480, 152, 267, 269, 481, 280, 312, 344, 482, 408, 440, 472, 483, 134, 142, 150, 484, 153, 174, 301, 485, 281, 313, 345, 486, 409, 441, 473, 487, 262, 270, 278, 488, 154, 302, 310, 489, 282, 314, 346, 490, 410, 442, 474, 491, 390, 398, 406, 492, 155, 430, 438, 493, 283, 315, 347, 494, 411, 443, 475, 495, 7, 15, 389, 496, 156, 395, 397, 497, 284, 316, 348, 498, 412, 444, 476, 499, 135, 143, 151, 500, 157, 175, 429, 501, 285, 317, 349, 502, 413, 445, 477, 503, 263, 271, 279, 504, 158, 303, 311, 505, 286, 318, 350, 506, 414, 446, 478, 507, 391, 399, 407, 508, 159, 431, 439, 509, 287, 319, 351, 510, 415, 447, 479, 511};

    for (int i = 0; i < 512; ++i) {
      if (ptr[i] != expected[i]) {
        std::cout <<" failed\n";
        std::ostream_iterator<int> Iter(std::cout, ", ");
        std::copy(ptr, ptr + 512, Iter);
        std::cout << std::endl;
        return false;
      }
    }
  
    std::cout <<" pass\n";
    return true;
     
  }
}


int main() {
  
  return !(test_load_blocked_striped<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT>() && test_load_blocked_striped<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED>() && test_load_subgroup_striped_standalone() && 
  test_load_blocked_striped_standalone<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED>() && test_load_blocked_striped_standalone<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT>());
}
