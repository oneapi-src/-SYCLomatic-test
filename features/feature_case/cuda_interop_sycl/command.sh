
export CUDA_HOME=/usr/local/cuda-11.4
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.4/bin:$PATH
export CUDA_INCLUDE_PATH=/usr/local/cuda-11.4/include
export SYCL_ROOT_DIR=/nfs/shm/proj/icl/cmplrarch/deploy_syclos/llorgsyclngefi2linux/20230911_160000/build/linux_qa_release
export CUDA_ROOT_DIR=/usr/local/cuda-11.4

clang++ -std=c++17 --cuda-gpu-arch=sm_70 -I${SYCL_ROOT_DIR}/include/ -I${SYCL_ROOT_DIR}/include/sycl/ -Wno-linker-warnings  -g test.cu -L${SYCL_ROOT_DIR}/lib -lOpenCL -lsycl -L${CUDA_ROOT_DIR}/lib64 -lcudart -o usm_vec_add.o
clang++ -std=c++17 --cuda-gpu-arch=sm_70 -I${SYCL_ROOT_DIR}/include/ -I${SYCL_ROOT_DIR}/include/sycl/ -I${CUDA_ROOT_DIR}/include -Wno-linker-warnings -xcuda -g test.cpp -L${SYCL_ROOT_DIR}/lib -lOpenCL -lsycl -L${CUDA_ROOT_DIR}/lib64 -lcudart  -o usm_vec_add_cpp.o

clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70  -L${CUDA_ROOT_DIR}/lib64 -xcuda test.cpp -lcudart -lcuda
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70  -L${CUDA_ROOT_DIR}/lib64 -lcudart -lcuda -xcuda test.cpp -o sycl.o

clang++ -fsycl  -I${CUDA_ROOT_DIR}/include --cuda-gpu-arch=sm_70 -L${CUDA_ROOT_DIR}/lib64 -xcuda test_sycl.cpp -lcudart -lcuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
