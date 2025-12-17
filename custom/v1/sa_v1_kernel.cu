#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, #x " must be float16")

// Q,K,V: [B,H,N,D] half
// O:     [B,H,N,D] half
// one block computes one query row (b,h,i) with multiple threads over D
// iterate keys in tiles of TK, load K,V into shared memory
// online softmax: maintain m (max), l (sum exp), and acc[D] for output
template<int D, int TK>
__global__ void sa_forward_v1_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int N, float scale
) {
    // grid: (B*H, N, 1)
    int bh = blockIdx.x;
    int i  = blockIdx.y;
    int b = bh / H;
    int h = bh - b * H;

    // thread over D
    int tid = threadIdx.x; // [0, D)
    if (tid >= D) return;

    // base pointers for this (b,h)
    // layout contiguous: (((b*H + h)*N + n)*D + d)
    int64_t base_bh = ((int64_t)b * H + h);
    const half* Q_bh = Q + base_bh * (int64_t)N * D;
    const half* K_bh = K + base_bh * (int64_t)N * D;
    const half* V_bh = V + base_bh * (int64_t)N * D;
    half* O_bh = O + base_bh * (int64_t)N * D;

    // load q_i[d] into register (float)
    float qd = __half2float(Q_bh[(int64_t)i * D + tid]);

    // shared memory for a tile of K and V: [TK, D]
    extern __shared__ half smem[];
    half* sK = smem;                 // TK*D
    half* sV = smem + TK * D;        // TK*D

    // online softmax states for this (i, tid)
    float m = -INFINITY;  // running max
    float l = 0.f;        // running sum exp
    float acc = 0.f;      // running output accumulator for this d

    // iterate over keys in tiles
    for (int t0 = 0; t0 < N; t0 += TK) {
        int tk = min(TK, N - t0);

        // cooperative load K/V tile into shared
        // each thread loads one d for many k rows (strided)
        for (int kk = 0; kk < tk; ++kk) {
            sK[kk * D + tid] = K_bh[(int64_t)(t0 + kk) * D + tid];
            sV[kk * D + tid] = V_bh[(int64_t)(t0 + kk) * D + tid];
        }
        __syncthreads();

        // First: compute scores for this tile and update online softmax + acc
        // We need score_j = dot(Q_i, K_j)*scale. Each thread has only q[d].
        // We'll do reduction over D for each key j using shared memory reduction style:
        // - compute partial = qd * K_j[d] per thread
        // - reduce across threads to get dot
        // For simplicity (v1), we use warp-level reduction assuming D=64 and block=64.
        // This is a medium-optimization baseline.

        for (int kk = 0; kk < tk; ++kk) {
            float kd = __half2float(sK[kk * D + tid]);
            float prod = qd * kd;

            // reduction across 64 threads -> dot
            // use warp shuffle (D=64 => 2 warps; do two-warp reduction via shared)
            __shared__ float red[2 * 64]; // enough
            // store per-thread
            red[tid] = prod;
            __syncthreads();

            // reduce within each warp
            float sum = red[tid];
            // warp 0: tid 0-31, warp 1: tid 32-63
            // do tree reduction in shared (simple and stable for v1)
            // reduce by half each step
            for (int offset = 32; offset > 0; offset >>= 1) {
                if (tid < offset) red[tid] += red[tid + offset];
                __syncthreads();
            }
            sum = red[0]; // only tid 0 has full sum
            __syncthreads();

            float score;
            if (tid == 0) red[0] = sum * scale;
            __syncthreads();
            score = red[0];
            __syncthreads();

            // online softmax update (per thread, same score broadcast)
            float m_new = fmaxf(m, score);
            float alpha = expf(m - m_new);
            float p = expf(score - m_new);

            // update acc for this d: acc = acc*alpha + p * V_j[d]
            float vd = __half2float(sV[kk * D + tid]);
            acc = acc * alpha + p * vd;

            // update l, m
            l = l * alpha + p;
            m = m_new;
        }

        __syncthreads();
    }

    // normalize
    acc = acc / l;

    // write output
    O_bh[(int64_t)i * D + tid] = __float2half(acc);
}

torch::Tensor sa_forward_v1(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_HALF(Q); CHECK_HALF(K); CHECK_HALF(V);

    auto B = (int)Q.size(0);
    auto H = (int)Q.size(1);
    auto N = (int)Q.size(2);
    auto D = (int)Q.size(3);

    TORCH_CHECK(D == 64, "v1 kernel currently assumes D=64 (ViT-Base head dim).");

    auto O = torch::empty_like(Q);

    dim3 grid(B * H, N, 1);
    dim3 block(64, 1, 1);

    constexpr int TK = 64; // tile keys
    size_t shmem = 2 * TK * 64 * sizeof(half); // K + V

    sa_forward_v1_kernel<64, TK><<<grid, block, shmem>>>(
        (half*)Q.data_ptr<at::Half>(),
        (half*)K.data_ptr<at::Half>(),
        (half*)V.data_ptr<at::Half>(),
        (half*)O.data_ptr<at::Half>(),
        B, H, N, (float)scale
    );

    return O;
}
