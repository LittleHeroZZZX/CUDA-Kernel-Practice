#include "cstdio"

#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define COUARSE_SIZE 4

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__global__ void tile_gemm_kernel(const float *A, // M x K
                                 const float *B, // K x N
                                 float *C,       // M x N
                                 int M, int N, int K) {
  __shared__ float A_s[BLOCK_SIZE * COUARSE_SIZE][TILE_SIZE];
  __shared__ float B_s[TILE_SIZE][BLOCK_SIZE * COUARSE_SIZE];

  int c_row = COUARSE_SIZE * (blockIdx.y * blockDim.y + threadIdx.y);
  int c_col = COUARSE_SIZE * (blockIdx.x * blockDim.x + threadIdx.x);
  if (c_row >= M || c_col >= N)
    return;
  float temp[COUARSE_SIZE][COUARSE_SIZE];
#pragma unroll
  for (int i = 0; i < COUARSE_SIZE * COUARSE_SIZE; i++) {
    ((float *)temp)[i] = 0;
  }
  for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; i++) {
    int tile_index = i * TILE_SIZE + threadIdx.x;
    A_s[threadIdx.y * 2][threadIdx.x] = A[c_row * K + tile_index];
    A_s[threadIdx.y * 2 + 1][threadIdx.x] = A[(c_row + 1) * K + tile_index];
    B_s[threadIdx.y][threadIdx.x * 2] =
        B[(i * TILE_SIZE + threadIdx.y) * N + c_col];
    B_s[threadIdx.y][threadIdx.x * 2 + 1] =
        B[(i * TILE_SIZE + threadIdx.y) * N + c_col + 1];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < TILE_SIZE; j++) {
      float a0 = A_s[threadIdx.y * 2][j]; // 显式存到寄存器
      float a1 = A_s[threadIdx.y * 2 + 1][j];
      float b0 = B_s[j][threadIdx.x * 2];
      float b1 = B_s[j][threadIdx.x * 2 + 1];
      temp[0][0] += a0 * b0;
      temp[0][1] += a0 * b1;
      temp[1][0] += a1 * b0;
      temp[1][1] += a1 * b1;
    }
    __syncthreads();
  }
  C[c_row * N + c_col] = temp[0][0];
  C[c_row * N + c_col + 1] = temp[0][1];
  C[(c_row + 1) * N + c_col] = temp[1][0];
  C[(c_row + 1) * N + c_col + 1] = temp[1][1];
}

void tile_gemm(float *A, float *B, float *C, int M, int N, int K) {
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, M * K * sizeof(float));
  cudaMalloc(&B_d, K * N * sizeof(float));
  cudaMalloc(&C_d, M * N * sizeof(float));

  cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyDefault);

  dim3 gridDim(ceil_div(N, BLOCK_SIZE * COUARSE_SIZE),
               ceil_div(M, BLOCK_SIZE * COUARSE_SIZE));
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
  // for (int i = 0; i < M; i++)
  //   for (int j = 0; j < N; j++) {
  //     float sum = 0.0f;
  //     for (int k = 0; k < K; k++)
  //       sum += A[i * K + k] * B[k * N + j];
  //     C_ref[i * N + j] = sum;
  //   }
  cudaEvent_t start, stop;
  for (int i = 0; i < 10; i++)
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
