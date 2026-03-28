#include "cstdio"

#define BLOCK_SIZE 32
#define TILE_SIZE 32

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__global__ void tile_gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  __shared__ float A_s[BLOCK_SIZE][TILE_SIZE];
  __shared__ float B_s[TILE_SIZE][BLOCK_SIZE];

  int c_row = blockIdx.y * blockDim.y + threadIdx.y;
  int c_col = blockIdx.x * blockDim.x + threadIdx.x;
  
  float temp = 0.0f;
  // 遍历所有的块
  for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; i++) {
    // 1. 安全加载 A 矩阵到共享内存
    int a_col = i * TILE_SIZE + threadIdx.x;
    if (c_row < M && a_col < K) {
      A_s[threadIdx.y][threadIdx.x] = A[c_row * K + a_col];
    } else {
      A_s[threadIdx.y][threadIdx.x] = 0.0f; // 越界补零
    }

    // 2. 安全加载 B 矩阵到共享内存
    int b_row = i * TILE_SIZE + threadIdx.y;
    if (b_row < K && c_col < N) {
      B_s[threadIdx.y][threadIdx.x] = B[b_row * N + c_col];
    } else {
      B_s[threadIdx.y][threadIdx.x] = 0.0f; // 越界补零
    }
    
    __syncthreads(); // 现在所有线程都会执行到这里，不会死锁了

    // 3. 计算乘加
    for (int j = 0; j < TILE_SIZE; j++) {
      temp += A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
    }
    __syncthreads();
  }

  // 4. 最后判断是否在输出范围内，再写回 C
  if (c_row < M && c_col < N) {
    C[c_row * N + c_col] = temp;
  }
}

void tile_gemm(float *A, float *B, float *C, int M, int N, int K) {
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, M * K * sizeof(float));
  cudaMalloc(&B_d, K * N * sizeof(float));
  cudaMalloc(&C_d, M * N * sizeof(float));

  cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyDefault);

  dim3 gridDim(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  tile_gemm_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);

  cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDefault);
}

int main() {
  // 在 main 里
  int M, N, K;
  M = N = K = 4096;

  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));

  // 填随机数
  for (int i = 0; i < M * K; i++)
    A[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < K * N; i++)
    B[i] = rand() / (float)RAND_MAX;

  float *C_ref = (float *)malloc(M * N * sizeof(float));
//   for (int i = 0; i < M; i++)
//     for (int j = 0; j < N; j++) {
//       float sum = 0.0f;
//       for (int k = 0; k < K; k++)
//         sum += A[i * K + k] * B[k * N + j];
//       C_ref[i * N + j] = sum;
//     }
  cudaEvent_t start, stop;
  for(int i=0; i<10; i++)
      tile_gemm(A, B, C, M, N, K);

  cudaDeviceSynchronize();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  tile_gemm(A, B, C, M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float max_diff = 0.0f;
  for (int i = 0; i < M * N; i++)
    max_diff = fmax(max_diff, fabs(C[i] - C_ref[i]));
  printf("Max diff: %e\n", max_diff);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Time: %.3f ms\n", ms);
}
