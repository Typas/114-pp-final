/**
 * Self-Attention v5 - Large Q tile for better memory efficiency
 *
 * Key optimizations over v4:
 * 1. Q_ROWS = 128 (vs 64 in v4) - reduces K/V redundant loads by 2x
 * 2. 8 warps per block for more parallelism
 * 3. Better arithmetic intensity: more computation per K/V load
 *
 * Memory efficiency analysis:
 * - For N=197: ceil(197/128) = 2 K/V loads vs 4 in v4
 * - For N=577: ceil(577/128) = 5 K/V loads vs 10 in v4
 * - Theoretical 2x reduction in memory traffic
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, #x " must be float16")

// ======= Tunables =======
constexpr int D = 64;
constexpr int D2 = 32;              // half2 lanes
constexpr int NUM_WARPS = 8;        // 8 warps per block (doubled from v4)
constexpr int BLOCK_SIZE = NUM_WARPS * 32;  // 256 threads
constexpr int Q_ROWS_PER_WARP = 16; // Each warp handles 16 query rows
constexpr int Q_ROWS = NUM_WARPS * Q_ROWS_PER_WARP;  // 128 query rows per block (doubled!)
constexpr int WM = 16;              // wmma M
constexpr int WN = 16;              // wmma N
constexpr int WK = 16;              // wmma K
constexpr int TK = 64;              // KV tile
// ========================

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__global__ void __launch_bounds__(256)
sa_forward_v5_kernel(
    const half* __restrict__ Q,  // [B,H,N,D]
    const half* __restrict__ K,  // [B,H,N,D]
    const half* __restrict__ V,  // [B,H,N,D]
    half* __restrict__ O,        // [B,H,N,D]
    int B, int H, int N,
    float scale
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int bh = blockIdx.x;        // 0..B*H-1
    const int q_tile = blockIdx.y;    // 0..ceil(N/Q_ROWS)-1

    // Each warp handles different Q rows
    const int q_base = q_tile * Q_ROWS + warp_id * Q_ROWS_PER_WARP;

    const int b = bh / H;
    const int h = bh % H;

    const int64_t base_bh = (int64_t)(b * H + h) * (int64_t)N * D;
    const half* Q_bh = Q + base_bh;
    const half* K_bh = K + base_bh;
    const half* V_bh = V + base_bh;
    half* O_bh = O + base_bh;

    // Dynamic shared memory layout:
    // Kc  : [64,64] half col_major for WMMA = 8KB
    // V2  : [64,32] half2 row_major = 8KB
    extern __shared__ unsigned char smem[];
    half*  Kc   = (half*)smem;                        // 64*64 = 8KB
    half2* V2   = (half2*)(Kc + D * TK);              // 64*32*4 = 8KB

    // Per-warp Q storage in static shared memory
    __shared__ half Qs[NUM_WARPS][WM * D];  // 8 * 16 * 64 = 16KB

    // Per-warp score buffer
    __shared__ float Sbuf[NUM_WARPS][WM * WN];  // 8 * 16 * 16 = 8KB

    // Online softmax state per query row (in registers)
    float m[Q_ROWS_PER_WARP];
    float lsum[Q_ROWS_PER_WARP];
    float2 acc[Q_ROWS_PER_WARP];

    #pragma unroll
    for (int qi = 0; qi < Q_ROWS_PER_WARP; ++qi) {
        m[qi] = -INFINITY;
        lsum[qi] = 0.f;
        acc[qi] = make_float2(0.f, 0.f);
    }

    // Each warp loads its own Q rows
    for (int idx = lane; idx < WM * D; idx += 32) {
        int r = idx / D;
        int c = idx % D;
        int q_idx = q_base + r;

        half v = __float2half(0.f);
        if (q_idx < N) {
            v = Q_bh[(int64_t)q_idx * D + c];
        }
        Qs[warp_id][r * D + c] = v;
    }
    __syncthreads();

    // Stream over KV tiles
    for (int t0 = 0; t0 < N; t0 += TK) {
        // Cooperative load K tile (all threads participate)
        for (int idx = tid; idx < D * TK; idx += BLOCK_SIZE) {
            int k = idx / TK;       // 0..63 (dimension)
            int j = idx % TK;       // 0..63 (key index in tile)
            int key_idx = t0 + j;

            half v = __float2half(0.f);
            if (key_idx < N) {
                v = K_bh[(int64_t)key_idx * D + k];
            }
            Kc[k + j * D] = v;  // col_major layout
        }

        // Cooperative load V tile as half2
        for (int idx = tid; idx < TK * D2; idx += BLOCK_SIZE) {
            int j  = idx / D2;      // 0..63
            int d2 = idx % D2;      // 0..31
            int key_idx = t0 + j;

            half2 v = __float2half2_rn(0.f);
            if (key_idx < N) {
                const half2* Vh2 = reinterpret_cast<const half2*>(V_bh + (int64_t)key_idx * D);
                v = Vh2[d2];
            }
            V2[j * D2 + d2] = v;
        }
        __syncthreads();

        // Process keys in chunks of 16
        #pragma unroll
        for (int n_tile = 0; n_tile < TK; n_tile += WN) {
            // WMMA: S(16x16) = Qs(16x64) * Ksub(64x16)
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;
            wmma::fill_fragment(c, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < D; k_tile += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> bfrag;

                const half* A_ptr = Qs[warp_id] + k_tile;         // ld=D
                const half* B_ptr = Kc + k_tile + n_tile * D;     // col_major, ld=D

                wmma::load_matrix_sync(a, A_ptr, D);
                wmma::load_matrix_sync(bfrag, B_ptr, D);
                wmma::mma_sync(c, a, bfrag, c);
            }

            // Store S to per-warp buffer
            wmma::store_matrix_sync(Sbuf[warp_id], c, WN, wmma::mem_row_major);
            __syncwarp();

            // Update online softmax and accumulate output
            #pragma unroll
            for (int col = 0; col < WN; ++col) {
                int j_local = n_tile + col;
                int key_idx = t0 + j_local;
                if (key_idx >= N) break;

                // V for this key and this lane
                half2 vv2 = V2[j_local * D2 + lane];
                float2 vf = make_float2(__half2float(__low2half(vv2)),
                                        __half2float(__high2half(vv2)));

                #pragma unroll
                for (int qi = 0; qi < Q_ROWS_PER_WARP; ++qi) {
                    int q_idx = q_base + qi;
                    if (q_idx >= N) continue;

                    float score = 0.f;
                    if (lane == 0) {
                        score = Sbuf[warp_id][qi * WN + col] * scale;
                    }
                    score = __shfl_sync(0xffffffff, score, 0);

                    float m_new = fmaxf(m[qi], score);
                    float alpha = fast_exp(m[qi] - m_new);
                    float p = fast_exp(score - m_new);

                    acc[qi].x = acc[qi].x * alpha + p * vf.x;
                    acc[qi].y = acc[qi].y * alpha + p * vf.y;
                    lsum[qi]  = lsum[qi] * alpha + p;
                    m[qi]     = m_new;
                }
            }
            __syncwarp();
        }
        __syncthreads();  // Before next tile load
    }

    // Write output half2
    half2* O2 = reinterpret_cast<half2*>(O_bh);
    #pragma unroll
    for (int qi = 0; qi < Q_ROWS_PER_WARP; ++qi) {
        int q_idx = q_base + qi;
        if (q_idx >= N) continue;

        float inv_l = (lsum[qi] > 0.f) ? (1.f / lsum[qi]) : 0.f;
        half2 out2 = __floats2half2_rn(acc[qi].x * inv_l, acc[qi].y * inv_l);
        O2[(int64_t)q_idx * D2 + lane] = out2;
    }
}

torch::Tensor sa_forward_v5(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_HALF(Q); CHECK_HALF(K); CHECK_HALF(V);

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int d = (int)Q.size(3);
    TORCH_CHECK(d == 64, "This v5 kernel assumes head dim D=64.");

    auto O = torch::empty_like(Q);

    int grid_y = (N + Q_ROWS - 1) / Q_ROWS;
    dim3 grid(B * H, grid_y, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    // Dynamic shared memory:
    // Kc: 64*64*2 = 8KB
    // V2: 64*32*4 = 8KB
    // Total dynamic: 16KB
    // Static shared:
    // Qs: 8*16*64*2 = 16KB
    // Sbuf: 8*16*16*4 = 8KB
    // Total static: 24KB
    // Grand total: ~40KB (within 48KB limit for most GPUs)
    size_t shmem =
        (64ull * 64 * sizeof(half)) +       // Kc
        (64ull * 32 * sizeof(half2));       // V2

    sa_forward_v5_kernel<<<grid, block, shmem>>>(
        (const half*)Q.data_ptr<at::Half>(),
        (const half*)K.data_ptr<at::Half>(),
        (const half*)V.data_ptr<at::Half>(),
        (half*)O.data_ptr<at::Half>(),
        B, H, N,
        (float)scale
    );

    return O;
}
