// ====------ thrust-merge_by_key.cu---------- *- CUDA -* ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

bool test_host() {
  for (auto overload=0; overload<4; overload++) {
    int  A_keys[16] = { 2,  4,  7, 13, 55, 65, 65, 66, 75, 84, 84, 84, 93, 94, 95, 97};
    int  A_vals[16] = {12, 12, 13, 14, 52,  1,  0,  7,  5, 45, -1, -4, 21, 71, 99, -1};
    int  B_keys[18] = {13, 15, 21, 41, 41, 45, 57, 57, 60, 65, 70, 70, 84, 90, 90, 95, 97, 98};
    int  B_vals[18] = { 1, 10, 32, 20, 14,  2, 42,  3, 11,  0,  0, -1, 14, 23, 71, 99, -1, -2};

    // results validated with nvcc
    int expected_keys_result[] = {2, 4, 7, 13, 13, 15, 21, 41, 41, 45, 55, 57, 57, 60, 65, 65, 65, 66, 70, 70, 75, 84, 84, 84, 84, 90, 90, 93, 94, 95, 95, 97, 97, 98};
    int expected_vals_result[] = {12, 12, 13, 14, 1, 10, 32, 20, 14, 2, 52, 42, 3, 11, 1, 0, 0, 7, 0, -1, 5, 45, -1, -4, 14, 23, 71, 21, 71, 99, 99, -1, -1, -2};

    thrust::host_vector<int> h_A_keys(A_keys,A_keys+16);
    thrust::host_vector<int> h_A_vals(A_vals,A_vals+16);
    thrust::host_vector<int> h_B_keys(B_keys,B_keys+18);
    thrust::host_vector<int> h_B_vals(B_vals,B_vals+18);

    thrust::host_vector<int> keys_result(34,0);
    thrust::host_vector<int> vals_result(34,0);

    thrust::pair<thrust::host_vector<int>::iterator,thrust::host_vector<int>::iterator> end;
    end = (overload==0 ? thrust::merge_by_key(h_A_keys.begin(), h_A_keys.end(),
                                              h_B_keys.begin(), h_B_keys.end(),
                                              h_A_vals.begin(), h_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin())     :
           overload==1 ? thrust::merge_by_key(h_A_keys.begin(), h_A_keys.end(),
                                              h_B_keys.begin(), h_B_keys.end(),
                                              h_A_vals.begin(), h_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin(),
                                              thrust::less<int>())                          :
           overload==2 ? thrust::merge_by_key(thrust::host,
                                              h_A_keys.begin(), h_A_keys.end(),
                                              h_B_keys.begin(), h_B_keys.end(),
                                              h_A_vals.begin(), h_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin())     :
                         thrust::merge_by_key(thrust::host,
                                              h_A_keys.begin(), h_A_keys.end(),
                                              h_B_keys.begin(), h_B_keys.end(),
                                              h_A_vals.begin(), h_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin(),
                                              thrust::less<int>()));
           
    if (end.first!=keys_result.end() ||
        end.second!=vals_result.end()) {
      std::cerr << "end mismatch\n";
      return false;      
    }
  
    for (auto i=0; i<keys_result.size(); i++) {
      if (keys_result[i]!=expected_keys_result[i]) {
        std::cerr << "keys mismatch\n";    
        return false;
      }
    }

    for (auto i=0; i<vals_result.size(); i++) {
      if (vals_result[i]!=expected_vals_result[i]) {
        std::cerr << "vals mismatch\n";
        return false;
      }
    }
    std::cerr << "Test passed host " << overload << "\n";
  }
  
  return true;
}

bool test_device() {
  for (auto overload=0; overload<4; overload++) {
    int  A_keys[16] = { 2,  4,  7, 13, 55, 65, 65, 66, 75, 84, 84, 84, 93, 94, 95, 97};
    int  A_vals[16] = {12, 12, 13, 14, 52,  1,  0,  7,  5, 45, -1, -4, 21, 71, 99, -1};
    int  B_keys[18] = {13, 15, 21, 41, 41, 45, 57, 57, 60, 65, 70, 70, 84, 90, 90, 95, 97, 98};
    int  B_vals[18] = { 1, 10, 32, 20, 14,  2, 42,  3, 11,  0,  0, -1, 14, 23, 71, 99, -1, -2};

    // results validated with nvcc
    int expected_keys_result[] = {2, 4, 7, 13, 13, 15, 21, 41, 41, 45, 55, 57, 57, 60, 65, 65, 65, 66, 70, 70, 75, 84, 84, 84, 84, 90, 90, 93, 94, 95, 95, 97, 97, 98};
    int expected_vals_result[] = {12, 12, 13, 14, 1, 10, 32, 20, 14, 2, 52, 42, 3, 11, 1, 0, 0, 7, 0, -1, 5, 45, -1, -4, 14, 23, 71, 21, 71, 99, 99, -1, -1, -2};

    thrust::device_vector<int> d_A_keys(A_keys,A_keys+16);
    thrust::device_vector<int> d_A_vals(A_vals,A_vals+16);
    thrust::device_vector<int> d_B_keys(B_keys,B_keys+18);
    thrust::device_vector<int> d_B_vals(B_vals,B_vals+18);

    thrust::device_vector<int> keys_result(34,0);
    thrust::device_vector<int> vals_result(34,0);

    thrust::pair<thrust::device_vector<int>::iterator,thrust::device_vector<int>::iterator> end;
    end = (overload==0 ? thrust::merge_by_key(d_A_keys.begin(), d_A_keys.end(),
                                              d_B_keys.begin(), d_B_keys.end(),
                                              d_A_vals.begin(), d_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin())     :
           overload==1 ? thrust::merge_by_key(d_A_keys.begin(), d_A_keys.end(),
                                              d_B_keys.begin(), d_B_keys.end(),
                                              d_A_vals.begin(), d_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin(),
                                              thrust::less<int>())                          :
           overload==2 ? thrust::merge_by_key(thrust::device,
                                              d_A_keys.begin(), d_A_keys.end(),
                                              d_B_keys.begin(), d_B_keys.end(),
                                              d_A_vals.begin(), d_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin())     :
                         thrust::merge_by_key(thrust::device,
                                              d_A_keys.begin(), d_A_keys.end(),
                                              d_B_keys.begin(), d_B_keys.end(),
                                              d_A_vals.begin(), d_B_vals.begin(),
                                              keys_result.begin(), vals_result.begin(),
                                              thrust::less<int>()));
           
    if (end.first!=keys_result.end() ||
        end.second!=vals_result.end()) {
      std::cerr << "end mismatch\n";
      return false;      
    }
  
    for (auto i=0; i<keys_result.size(); i++) {
      if (keys_result[i]!=expected_keys_result[i]) {
        std::cerr << "keys mismatch\n";    
        return false;
      }
    }

    for (auto i=0; i<vals_result.size(); i++) {
      if (vals_result[i]!=expected_vals_result[i]) {
        std::cerr << "vals mismatch\n";
        return false;
      }
    }
    std::cerr << "Test passed device " << overload << "\n";
  }

  return true;
}

template<typename KEY_TYPE, typename VAL_TYPE>
bool test_host() {
  for (auto overload=0; overload<4; overload++) {
    KEY_TYPE  A_keys[16] = { 2,  4,  7, 13, 55, 65, 65, 66, 75, 84, 84, 84, 93, 94, 95, 97};
    VAL_TYPE  A_vals[16] = {12, 12, 13, 14, 52,  1,  0,  7,  5, 45, -1, -4, 21, 71, 99, -1};
    KEY_TYPE  B_keys[18] = {13, 15, 21, 41, 41, 45, 57, 57, 60, 65, 70, 70, 84, 90, 90, 95, 97, 98};
    VAL_TYPE  B_vals[18] = { 1, 10, 32, 20, 14,  2, 42,  3, 11,  0,  0, -1, 14, 23, 71, 99, -1, -2};

    // results validated with nvcc
    KEY_TYPE expected_keys_result[] = {2, 4, 7, 13, 13, 15, 21, 41, 41, 45, 55, 57, 57, 60, 65, 65, 65, 66, 70, 70, 75, 84, 84, 84, 84, 90, 90, 93, 94, 95, 95, 97, 97, 98};
    VAL_TYPE expected_vals_result[] = {12, 12, 13, 14, 1, 10, 32, 20, 14, 2, 52, 42, 3, 11, 1, 0, 0, 7, 0, -1, 5, 45, -1, -4, 14, 23, 71, 21, 71, 99, 99, -1, -1, -2};

    thrust::host_vector<KEY_TYPE> h_A_keys(A_keys,A_keys+16);
    thrust::host_vector<VAL_TYPE> h_A_vals(A_vals,A_vals+16);
    thrust::host_vector<KEY_TYPE> h_B_keys(B_keys,B_keys+18);
    thrust::host_vector<VAL_TYPE> h_B_vals(B_vals,B_vals+18);

    thrust::host_vector<KEY_TYPE> keys_result(34,0);
    thrust::host_vector<VAL_TYPE> vals_result(34,0);

    auto end = (overload==0 ? thrust::merge_by_key(h_A_keys.begin(), h_A_keys.end(),
                                                   h_B_keys.begin(), h_B_keys.end(),
                                                   h_A_vals.begin(), h_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin())     :
                overload==1 ? thrust::merge_by_key(h_A_keys.begin(), h_A_keys.end(),
                                                   h_B_keys.begin(), h_B_keys.end(),
                                                   h_A_vals.begin(), h_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin(),
                                                   thrust::less<KEY_TYPE>())                     :
                overload==2 ? thrust::merge_by_key(thrust::host,
                                                   h_A_keys.begin(), h_A_keys.end(),
                                                   h_B_keys.begin(), h_B_keys.end(),
                                                   h_A_vals.begin(), h_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin())     :
                              thrust::merge_by_key(thrust::host,
                                                   h_A_keys.begin(), h_A_keys.end(),
                                                   h_B_keys.begin(), h_B_keys.end(),
                                                   h_A_vals.begin(), h_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin(),
                                                   thrust::less<KEY_TYPE>()));
           
    if (end.first!=keys_result.end() ||
        end.second!=vals_result.end()) {
      std::cerr << "end mismatch\n";
      return false;      
    }
  
    for (auto i=0; i<keys_result.size(); i++) {
      if (keys_result[i]!=expected_keys_result[i]) {
        std::cerr << "keys mismatch\n";    
        return false;
      }
    }

    for (auto i=0; i<vals_result.size(); i++) {
      if (vals_result[i]!=expected_vals_result[i]) {
        std::cerr << "vals mismatch\n";
        return false;
      }
    }
    std::cerr << "Test passed host " << overload << " " << __PRETTY_FUNCTION__ << "\n";
  }
  
  return true;
}

template<typename KEY_TYPE, typename VAL_TYPE>
bool test_device() {
  for (auto overload=0; overload<4; overload++) {
    KEY_TYPE  A_keys[16] = { 2,  4,  7, 13, 55, 65, 65, 66, 75, 84, 84, 84, 93, 94, 95, 97};
    VAL_TYPE  A_vals[16] = {12, 12, 13, 14, 52,  1,  0,  7,  5, 45, -1, -4, 21, 71, 99, -1};
    KEY_TYPE  B_keys[18] = {13, 15, 21, 41, 41, 45, 57, 57, 60, 65, 70, 70, 84, 90, 90, 95, 97, 98};
    VAL_TYPE  B_vals[18] = { 1, 10, 32, 20, 14,  2, 42,  3, 11,  0,  0, -1, 14, 23, 71, 99, -1, -2};

    // results validated with nvcc
    KEY_TYPE expected_keys_result[] = {2, 4, 7, 13, 13, 15, 21, 41, 41, 45, 55, 57, 57, 60, 65, 65, 65, 66, 70, 70, 75, 84, 84, 84, 84, 90, 90, 93, 94, 95, 95, 97, 97, 98};
    VAL_TYPE expected_vals_result[] = {12, 12, 13, 14, 1, 10, 32, 20, 14, 2, 52, 42, 3, 11, 1, 0, 0, 7, 0, -1, 5, 45, -1, -4, 14, 23, 71, 21, 71, 99, 99, -1, -1, -2};

    thrust::device_vector<KEY_TYPE> d_A_keys(A_keys,A_keys+16);
    thrust::device_vector<VAL_TYPE> d_A_vals(A_vals,A_vals+16);
    thrust::device_vector<KEY_TYPE> d_B_keys(B_keys,B_keys+18);
    thrust::device_vector<VAL_TYPE> d_B_vals(B_vals,B_vals+18);

    thrust::device_vector<KEY_TYPE> keys_result(34,0);
    thrust::device_vector<VAL_TYPE> vals_result(34,0);

    auto end = (overload==0 ? thrust::merge_by_key(d_A_keys.begin(), d_A_keys.end(),
                                                   d_B_keys.begin(), d_B_keys.end(),
                                                   d_A_vals.begin(), d_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin())     :
                overload==1 ? thrust::merge_by_key(d_A_keys.begin(), d_A_keys.end(),
                                                   d_B_keys.begin(), d_B_keys.end(),
                                                   d_A_vals.begin(), d_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin(),
                                                   thrust::less<KEY_TYPE>())                     :
                overload==2 ? thrust::merge_by_key(thrust::device,
                                                   d_A_keys.begin(), d_A_keys.end(),
                                                   d_B_keys.begin(), d_B_keys.end(),
                                                   d_A_vals.begin(), d_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin())     :
                              thrust::merge_by_key(thrust::device,
                                                   d_A_keys.begin(), d_A_keys.end(),
                                                   d_B_keys.begin(), d_B_keys.end(),
                                                   d_A_vals.begin(), d_B_vals.begin(),
                                                   keys_result.begin(), vals_result.begin(),
                                                   thrust::less<KEY_TYPE>()));
           
    if (end.first!=keys_result.end() ||
        end.second!=vals_result.end()) {
      std::cerr << "end mismatch\n";
      return false;      
    }
  
    for (auto i=0; i<keys_result.size(); i++) {
      if (keys_result[i]!=expected_keys_result[i]) {
        std::cerr << "keys mismatch\n";    
        return false;
      }
    }

    for (auto i=0; i<vals_result.size(); i++) {
      if (vals_result[i]!=expected_vals_result[i]) {
        std::cerr << "vals mismatch\n";
        return false;
      }
    }
    std::cerr << "Test passed device " << overload << " " << __PRETTY_FUNCTION__ << "\n";
  }
  
  return true;
}

int main() {
  if (!test_host() ||
      !test_device())
    return EXIT_FAILURE;

  if (!test_host<char,char>()      ||
      !test_host<char,short int>() ||
      !test_host<short int,short int>() ||
      !test_host<short int,int>()       ||
      !test_host<int,int>()       ||
      !test_host<int,long>()      ||
      !test_host<long,long>()      ||
      !test_host<long long,long long>() ||
      !test_host<float,float>()     ||
      !test_host<double,double>())
    return EXIT_FAILURE;
    
  if (!test_device<char,char>()      ||
      !test_device<char,float>()     ||
      !test_device<short int,char>()      ||
      !test_device<short int,float>()     ||
      !test_device<int,char>()      ||
      !test_device<long,long>()      ||
      !test_device<long long,long long>() ||
      !test_device<float,int>() ||
      !test_host<double,double>())
    return EXIT_FAILURE;  
    
  return EXIT_SUCCESS;
}
