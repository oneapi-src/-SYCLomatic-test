#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

//global vars for cublaslt
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;
cublasHandle_t cublas_handle;



void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))


void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }



float* make_fixed_float(size_t n){
  float* arr = (float*)malloc(n * sizeof(float));
  for(int i=0;i<n;i++){
    arr[i] = 1.0f;
  }
  return arr;
}

void matmul_forward_cublaslt(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC){

  int has_bias = (bias!=NULL);
  int has_gelu =0;
  
  
  if((uintptr_t)bias % 16 !=0){
    printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
    exit(EXIT_FAILURE);
  }
  
  int returnedResults = 0;
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatmulPreference_t preference;
  cublasLtMatrixLayout_t inputLayout;
  cublasLtMatrixLayout_t weightLayout;
  cublasLtMatrixLayout_t biasLayout;
  cublasLtMatrixLayout_t outputLayout;
  cublasLtMatmulHeuristicResult_t heuristic;
  
  cublasOperation_t opNoTranspose = CUBLAS_OP_N;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
  if (has_bias && has_gelu) {
      epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
  } else if (has_bias) {
      epilogueBias = CUBLASLT_EPILOGUE_BIAS;
  } else if (has_gelu) {
      epilogueBias = CUBLASLT_EPILOGUE_GELU;
  }
  cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

  // define matrix layouts
  cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
  cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B*T, C));
  cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B*T, OC));
  cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

  // create a preference handle with specified max workspace
  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // find a suitable algorithm
  cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
      weightLayout, inputLayout, outputLayout, outputLayout,
      preference, 1, &heuristic, &returnedResults));
  if (returnedResults == 0) {
      printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
          B, T, C, OC, has_bias, has_gelu);
      exit(EXIT_FAILURE);
  }

  // call the matmul
  const float alpha = 1.0f, beta = 0.0f;
  cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
      &alpha, weight, weightLayout, inp, inputLayout, &beta,
      out, outputLayout, out, outputLayout, &heuristic.algo,
      cublaslt_workspace, cublaslt_workspace_size, 0));

  // cleanups
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}


void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC){
                    
                    
  matmul_forward_cublaslt(out, inp, weight, bias, B, T, C, OC);                    

}

void validate_results(const float* kernel_result, int num_elements) {
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (kernel_result[i] == 769.0f) {
            printf("%f %f\n", kernel_result[i]);
        }
        else{
            nfaults++;
            if (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    if (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}


int main(int argc, char **argv) {
    srand(0);

    int B = 32;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_fixed_float(B * T * C);
    float* weight = make_fixed_float(OC * C);
    float* bias = make_fixed_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));
    matmul_forward(out, inp, weight, bias, B, T, C, OC);   
    validate_results(out, B * T * OC);
    
    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}
