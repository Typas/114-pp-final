#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, #x " must be float16")

// Fixed for ViT-Base head dim
constexpr int D  = 64;
constexpr int D2 = 32;   // half2 lanes
constexpr int TK = 64;   // KV tile size
constexpr int Q_ROWS = 8; // ★ v2.5: one block computes multiple query rows

__device__ __forceinline__ float warp_reduce_sum(float x) {
    // full warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__global__ void sa_forward_v2_kernel(
    const half* __restrict__ Q,  // [B,H,N,D]
    const half* __restrict__ K,  // [B,H,N,D]
    const half* __restrict__ V,  // [B,H,N,D]
    half* __restrict__ O,        // [B,H,N,D]
    int B, int H, int N,
    float scale
) {
    // one warp block
    int lane = threadIdx.x; // 0..31
    if (lane >= 32) return;

    int bh = blockIdx.x;    // 0..B*H-1
    int q_tile_id = blockIdx.y; // 0..ceil(N/Q_ROWS)-1
    int q_base = q_tile_id * Q_ROWS;

    int b = bh / H;
    int h = bh - b * H;

    int64_t base_bh = (int64_t)(b * H + h) * (int64_t)N * D;

    const half* Q_bh = Q + base_bh;
    const half* K_bh = K + base_bh;
    const half* V_bh = V + base_bh;
    half* O_bh = O + base_bh;

    // load Q tile into registers (half2 per lane)
    float2 qf[Q_ROWS];
    bool q_valid[Q_ROWS];

    #pragma unroll
    for (int qi = 0; qi < Q_ROWS; ++qi) {
        int q_idx = q_base + qi;
        q_valid[qi] = (q_idx < N);
        if (q_valid[qi]) {
            const half2* Q2 = reinterpret_cast<const half2*>(Q_bh + (int64_t)q_idx * D);
            half2 q2 = Q2[lane];
            qf[qi] = make_float2(__half2float(__low2half(q2)), __half2float(__high2half(q2)));
        } else {
            qf[qi] = make_float2(0.f, 0.f);
        }
    }

    // shared memory tiles: K and V, stored as half2 [TK, D2]
    extern __shared__ half2 smem2[];
    half2* sK = smem2;                 // TK*D2
    half2* sV = smem2 + TK * D2;       // TK*D2

    // online softmax state per query row
    float m[Q_ROWS];
    float l[Q_ROWS];
    float2 acc[Q_ROWS];

    #pragma unroll
    for (int qi = 0; qi < Q_ROWS; ++qi) {
        m[qi] = -INFINITY;
        l[qi] = 0.f;
        acc[qi] = make_float2(0.f, 0.f);
    }

    // loop over KV tiles
    for (int t0 = 0; t0 < N; t0 += TK) {
        // load K/V tile into shared
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            int j = t0 + kk;
            if (j < N) {
                const half2* K2 = reinterpret_cast<const half2*>(K_bh + (int64_t)j * D);
                const half2* V2 = reinterpret_cast<const half2*>(V_bh + (int64_t)j * D);
                sK[kk * D2 + lane] = K2[lane];
                sV[kk * D2 + lane] = V2[lane];
            } else {
                sK[kk * D2 + lane] = __float2half2_rn(0.f);
                sV[kk * D2 + lane] = __float2half2_rn(0.f);
            }
        }
        __syncthreads();

        // compute this tile
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            int j = t0 + kk;
            if (j >= N) break;

            half2 k2 = sK[kk * D2 + lane];
            float2 kf = make_float2(__half2float(__low2half(k2)), __half2float(__high2half(k2)));

            half2 v2 = sV[kk * D2 + lane];
            float2 vf = make_float2(__half2float(__low2half(v2)), __half2float(__high2half(v2)));

            // update each query row in this Q tile
            #pragma unroll
            for (int qi = 0; qi < Q_ROWS; ++qi) {
                if (!q_valid[qi]) continue;

                float partial = qf[qi].x * kf.x + qf[qi].y * kf.y;
                float dot = warp_reduce_sum(partial);
                float score = __shfl_sync(0xffffffff, dot, 0) * scale;

                // online softmax (stable)
                float m_new = fmaxf(m[qi], score);
                float alpha = expf(m[qi] - m_new);
                float p = expf(score - m_new);

                acc[qi].x = acc[qi].x * alpha + p * vf.x;
                acc[qi].y = acc[qi].y * alpha + p * vf.y;
                l[qi] = l[qi] * alpha + p;
                m[qi] = m_new;
            }
        }

        __syncthreads();
    }

    // write outputs
    half2* O2 = reinterpret_cast<half2*>(O_bh);

    #pragma unroll
    for (int qi = 0; qi < Q_ROWS; ++qi) {
        int q_idx = q_base + qi;
        if (!q_valid[qi]) continue;

        float inv_l = (l[qi] > 0.f) ? (1.f / l[qi]) : 0.f;
        float ox = acc[qi].x * inv_l;
        float oy = acc[qi].y * inv_l;

        half2 out2 = __floats2half2_rn(ox, oy);
        O2[(int64_t)q_idx * D2 + lane] = out2;
    }
}

torch::Tensor sa_forward_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_HALF(Q); CHECK_HALF(K); CHECK_HALF(V);

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int d = (int)Q.size(3);
    TORCH_CHECK(d == 64, "v2.5 kernel assumes head dim D=64.");

    auto O = torch::empty_like(Q);

    // grid.y now tiles queries by Q_ROWS
    int grid_y = (N + Q_ROWS - 1) / Q_ROWS;
    dim3 grid(B * H, grid_y, 1);
    dim3 block(32, 1, 1);  // one warp

    // shared: K and V tiles in half2
    size_t shmem = 2ull * TK * D2 * sizeof(half2); // 2 * 64 * 32 * 4 = 16384 bytes

    sa_forward_v2_kernel<<<grid, block, shmem>>>(
        (const half*)Q.data_ptr<at::Half>(),
        (const half*)K.data_ptr<at::Half>(),
        (const half*)V.data_ptr<at::Half>(),
        (half*)O.data_ptr<at::Half>(),
        B, H, N,
        (float)scale
    );

    return O;
}
