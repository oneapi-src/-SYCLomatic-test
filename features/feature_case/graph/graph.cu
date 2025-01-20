// ===------- graph.cu ------------------------------------ *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <vector>

const int blockSize = 256;
const int numBlocks = (10 + blockSize - 1) / blockSize;

__global__ void init(float *a) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < 10) {
    a[id] = 1.0f;
  }
}

__global__ void incrementA(float *a) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < 10) {
    a[id] += 1.0f;
  }
}

int main() {

  cudaGraph_t graph;

  cudaStream_t stream;

  cudaStreamCreate(&stream);

  float *d_a, h_a[10];

  cudaMalloc(&d_a, 10 * sizeof(float));

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  init<<<numBlocks, blockSize, 0, stream>>>(d_a);

  incrementA<<<numBlocks, blockSize, 0, stream>>>(d_a);

  cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t execGraph;
  cudaGraphInstantiate(&execGraph, graph, NULL, NULL, 0);

  cudaGraphLaunch(execGraph, stream);

  cudaStreamSynchronize(stream); // Ensure the graph has completed execution

  cudaMemcpy(h_a, d_a, 10 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    if (h_a[i] != 2.0f) {
      printf("Results do not match\n");
      return -1;
    }
  }

  // Get all nodes in the graph
  size_t numNodes;
  cudaGraphGetNodes(graph, nullptr, &numNodes);
  std::vector<cudaGraphNode_t> nodes(numNodes);
  cudaGraphGetNodes(graph, nodes.data(), &numNodes);

  // Get root nodes in the graph
  size_t numRootNodes;
  cudaGraphGetRootNodes(graph, nullptr, &numRootNodes);
  std::vector<cudaGraphNode_t> rootNodes(numRootNodes);
  cudaGraphGetRootNodes(graph, rootNodes.data(), &numRootNodes);

  if (numNodes != 2 || numRootNodes != 1) {
    printf("Number of nodes or root nodes do not match\n");
    return -1;
  }

  printf("Passed\n");

  cudaStreamDestroy(stream);
  cudaFree(d_a);
  cudaGraphExecDestroy(execGraph);

  return 0;
}
