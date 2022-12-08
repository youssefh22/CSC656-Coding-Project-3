#include <iostream>
#include <math.h>
#include <chrono>

// function to add the elements of two arrays
// CUDA Kernel function to add the elemnets of two arrays on the GPU
__global__ void add(int n, float *x, float *y)
{
  // grid is a collection of thread blocks, each of which is a collection of threads
  // block represents a group of threads that can be executed serially or in parrallel
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+= stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<24;  // 20 -> 24

  float *x, *y;
  // cudaMallocManaged allocate data in unified memory, returns a pointer that you can access from host code or device code
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int deviceID=0;

  cudaMemPrefetchAsync((void *)x, N*sizeof(float), deviceID);
  cudaMemPrefetchAsync((void *)y, N*sizeof(float), deviceID);

  // thread blocks are a collection of threads. All the threads in any single block can communicate
  // blockSize is the number of threads
  // numBlocks is the number of blocks
  int blockSize = 256;
  
  int numBlocks = (N + blockSize - 1) / blockSize;
  printf("thread blocks: %d\n", numBlocks);
  add<<<numBlocks, blockSize>>>(N, x, y);
  // wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}