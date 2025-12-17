#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float half_to_float(half x) { return __half2float(x); }
__device__ __forceinline__ half  float_to_half(float x) { return __float2half_rn(x); }

// Q,K,V: [B,H,N,D] contiguous in row-major
// index mapping: (((b*H + h)*N + i)*D + d)
__global__ void sa_forward_v0_kernel(const half* __restrict__ Q,
                                     const half* __restrict__ K,
                                     const half* __restrict__ V,
                                     half* __restrict__ O,
                                     int B, int H, int N, int D,
                                     float scale) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int i = blockIdx.z;

  int d = threadIdx.x; // 0..D-1 (we'll launch D threads)
  if (d >= D) return;

  // Pointer offsets for this (b,h,i)
  int base_q = (((b * H + h) * N + i) * D);

  // 1) compute max over j: scores_j = (Q_i · K_j)*scale
  float max_score = -1e20f;

  for (int j = 0; j < N; ++j) {
    int base_k = (((b * H + h) * N + j) * D);
    float dot = 0.f;
    // dot product over D
    for (int kk = 0; kk < D; ++kk) {
      float qv = half_to_float(Q[base_q + kk]);
      float kv = half_to_float(K[base_k + kk]);
      dot += qv * kv;
    }
    float score = dot * scale;
    if (score > max_score) max_score = score;
  }

  // 2) compute sum exp
  float sum_exp = 0.f;
  for (int j = 0; j < N; ++j) {
    int base_k = (((b * H + h) * N + j) * D);
    float dot = 0.f;
    for (int kk = 0; kk < D; ++kk) {
      float qv = half_to_float(Q[base_q + kk]);
      float kv = half_to_float(K[base_k + kk]);
      dot += qv * kv;
    }
    float score = dot * scale;
    sum_exp += __expf(score - max_score);
  }
  float inv_sum = 1.f / (sum_exp + 1e-9f);

  // 3) output O[b,h,i,d] = sum_j p_j * V[b,h,j,d]
  float out = 0.f;
  for (int j = 0; j < N; ++j) {
    int base_k = (((b * H + h) * N + j) * D);
    float dot = 0.f;
    for (int kk = 0; kk < D; ++kk) {
      float qv = half_to_float(Q[base_q + kk]);
      float kv = half_to_float(K[base_k + kk]);
      dot += qv * kv;
    }
    float score = dot * scale;
    float p = __expf(score - max_score) * inv_sum;

    int base_v = (((b * H + h) * N + j) * D);
    float vv = half_to_float(V[base_v + d]);
    out += p * vv;
  }

  int base_o = (((b * H + h) * N + i) * D);
  O[base_o + d] = float_to_half(out);
}

torch::Tensor sa_forward_cuda(torch::Tensor Q,
                              torch::Tensor K,
                              torch::Tensor V,
                              double scale) {
  const auto B = (int)Q.size(0);
  const auto H = (int)Q.size(1);
  const auto N = (int)Q.size(2);
  const auto D = (int)Q.size(3);

  auto O = torch::empty_like(Q);

  dim3 grid(B, H, N);
  dim3 block(256); // we'll guard d>=D
  // Better: block.x = nextPow2(D) but keep simple for v0

  sa_forward_v0_kernel<<<grid, block>>>(
      (half*)Q.data_ptr<at::Half>(),
      (half*)K.data_ptr<at::Half>(),
      (half*)V.data_ptr<at::Half>(),
      (half*)O.data_ptr<at::Half>(),
      B, H, N, D, (float)scale
  );

  return O;
}
