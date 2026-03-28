#include "cstdio"

#define BLOCK_SIZE 32

__global__ void native_gemm_kernel(const float *A, // M x K
                                   const float *B, // K x N
                                   float *C,       // M x N
                                   int M, int N, int K) {
  int c_row = blockIdx.y * blockDim.y + threadIdx.y;
  int c_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (c_row >= M || c_col >= N)
    return;

  float temp = 0.0f;
  for (int i = 0; i < K; i++) {
    temp += A[c_row * K + i] * B[i * N + c_col];
  }
  C[c_row * N + c_col] = temp;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

void native_gemm(float *A, float *B, float *C, int M, int N, int K) {
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, M * K * sizeof(float));
  cudaMalloc(&B_d, K * N * sizeof(float));
  cudaMalloc(&C_d, M * N * sizeof(float));

  cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyDefault);

  dim3 gridDim(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  native_gemm_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);

  cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDefault);
}

int main() {
  // 在 main 里
  int M = 4096, N = 4096, K = 4096;

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
// for(int i=0; i<10; i++)
//     native_gemm(A, B, C, M, N, K);

cudaDeviceSynchronize();
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
  native_gemm(A, B, C, M, N, K);
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
