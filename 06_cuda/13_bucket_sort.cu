#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __managed__ int sum, *bucket;

__global__ void zero(int *a) {
  a[threadIdx.x] = 0;
}

__global__ void atomic(int *a, int *key) {
  int n = key[threadIdx.x];
  atomicAdd(&a[n], 1);
}

__global__ void thread(int *key, int n) {
  key[sum + threadIdx.x] = n;
  __syncthreads();
  atomicAdd(&sum, 1);
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaMallocManaged(&bucket, range*sizeof(int));
  zero<<<1,range>>>(bucket);
  cudaDeviceSynchronize();

  int *keyArray;
  cudaMallocManaged(&keyArray, n*sizeof(int));
  std::copy(key.begin(), key.end(), keyArray);
  atomic<<<1,n>>>(bucket, keyArray);
  cudaDeviceSynchronize();

  for (int i=0; i<range; i++) {
    thread<<<1,bucket[i]>>>(keyArray, i);
    cudaDeviceSynchronize();
  }

  cudaFree(bucket);

  for (int i=0; i<n; i++) {
    printf("%d ",keyArray[i]);
  }
  printf("\n");
  cudaFree(keyArray);
}
