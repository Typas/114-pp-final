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
constexpr int D2 = 32;          // half2 lanes
constexpr int Q_ROWS = 16;      // block computes 16 query rows (WMMA)
constexpr int WM = 16;          // wmma M
constexpr int WN = 16;          // wmma N
constexpr int WK = 16;          // wmma K
constexpr int TK = 64;          // KV tile (must be multiple of 16)
// ========================

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__global__ void sa_forward_v3_kernel(
    const half* __restrict__ Q,  // [B,H,N,D]
    const half* __restrict__ K,  // [B,H,N,D]
    const half* __restrict__ V,  // [B,H,N,D]
    half* __restrict__ O,        // [B,H,N,D]
    int B, int H, int N,
    float scale
) {
    // 1 warp per block
    int lane = threadIdx.x;
    if (lane >= 32) return;

    int bh = blockIdx.x;        // 0..B*H-1
    int q_tile = blockIdx.y;    // 0..ceil(N/Q_ROWS)-1
    int q_base = q_tile * Q_ROWS;

    int b = bh / H;
    int h = bh - b * H;

    int64_t base_bh = (int64_t)(b * H + h) * (int64_t)N * D;
    const half* Q_bh = Q + base_bh;
    const half* K_bh = K + base_bh;
    const half* V_bh = V + base_bh;
    half* O_bh = O + base_bh;

    // Shared memory layout:
    // Qs  : [16,64] half row_major, ld=D
    // Kc  : [64,64] half col_major, stored as Kc[row + col*D] with ld=D
    // V2  : [64,32] half2 row_major, ld=D2
    // Sbuf: [16,16] float row_major, ld=16
    extern __shared__ unsigned char smem[];
    half*  Qs   = (half*)smem;                        // 16*64
    half*  Kc   = (half*)(Qs + WM * D);              // 64*64
    half2* V2   = (half2*)(Kc + D * TK);             // 64*32 half2
    float* Sbuf = (float*)(V2 + TK * D2);            // 16*16 float

    // Online softmax state per query row (only Q_ROWS rows are real)
    float m[Q_ROWS];
    float lsum[Q_ROWS];
    float2 acc[Q_ROWS];

    #pragma unroll
    for (int qi = 0; qi < Q_ROWS; ++qi) {
        m[qi] = -INFINITY;
        lsum[qi] = 0.f;
        acc[qi] = make_float2(0.f, 0.f);
    }

    // Load Q tile once (pad to 16 rows)
    for (int idx = lane; idx < WM * D; idx += 32) {
        int r = idx / D;
        int c = idx - r * D;
        int q_idx = q_base + r;

        half v = __float2half(0.f);
        if (r < Q_ROWS && q_idx < N) {
            v = Q_bh[(int64_t)q_idx * D + c];
        }
        Qs[r * D + c] = v;
    }
    __syncwarp();

    // Stream over KV tiles
    for (int t0 = 0; t0 < N; t0 += TK) {

        // Load K tile into Kc (col_major, ld=D): element (row=k, col=j) -> Kc[k + j*D]
        for (int idx = lane; idx < D * TK; idx += 32) {
            int k = idx / TK;       // 0..63
            int j = idx - k * TK;   // 0..63
            int key_idx = t0 + j;

            half v = __float2half(0.f);
            if (key_idx < N) {
                v = K_bh[(int64_t)key_idx * D + k];
            }
            Kc[k + j * D] = v;
        }

        // Load V tile as half2 [j, d2]
        for (int idx = lane; idx < TK * D2; idx += 32) {
            int j  = idx / D2;      // 0..63
            int d2 = idx - j * D2;  // 0..31
            int key_idx = t0 + j;

            half2 v = __float2half2_rn(0.f);
            if (key_idx < N) {
                const half2* Vh2 = reinterpret_cast<const half2*>(V_bh + (int64_t)key_idx * D);
                v = Vh2[d2];
            }
            V2[j * D2 + d2] = v;
        }
        __syncwarp();

        // keys in chunks of 16
        #pragma unroll
        for (int n_tile = 0; n_tile < TK; n_tile += WN) {

            // WMMA: S(16x16) = Qs(16x64) * Ksub(64x16)
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;
            wmma::fill_fragment(c, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < D; k_tile += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> bfrag;

                const half* A_ptr = Qs + k_tile;                 // ld=D
                const half* B_ptr = Kc + k_tile + n_tile * D;    // col_major, ld=D

                wmma::load_matrix_sync(a, A_ptr, D);
                wmma::load_matrix_sync(bfrag, B_ptr, D);
                wmma::mma_sync(c, a, bfrag, c);
            }

            // store S to shared float[16x16]
            wmma::store_matrix_sync(Sbuf, c, WN, wmma::mem_row_major);
            __syncwarp();

            // update online softmax and accumulate output
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
                for (int qi = 0; qi < Q_ROWS; ++qi) {
                    int q_idx = q_base + qi;
                    if (q_idx >= N) continue;

                    float score = 0.f;
                    if (lane == 0) {
                        score = Sbuf[qi * WN + col] * scale;
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
    }

    // write output half2
    half2* O2 = reinterpret_cast<half2*>(O_bh);
    #pragma unroll
    for (int qi = 0; qi < Q_ROWS; ++qi) {
        int q_idx = q_base + qi;
        if (q_idx >= N) continue;

        float inv_l = (lsum[qi] > 0.f) ? (1.f / lsum[qi]) : 0.f;
        half2 out2 = __floats2half2_rn(acc[qi].x * inv_l, acc[qi].y * inv_l);
        O2[(int64_t)q_idx * D2 + lane] = out2;
    }
}

torch::Tensor sa_forward_v3(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_HALF(Q); CHECK_HALF(K); CHECK_HALF(V);

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int d = (int)Q.size(3);
    TORCH_CHECK(d == 64, "This v3' kernel assumes head dim D=64.");

    auto O = torch::empty_like(Q);

    int grid_y = (N + Q_ROWS - 1) / Q_ROWS;
    dim3 grid(B * H, grid_y, 1);
    dim3 block(32, 1, 1);

    size_t shmem =
        (16ull * 64 * sizeof(half)) +     // Qs
        (64ull * 64 * sizeof(half)) +     // Kc
        (64ull * 32 * sizeof(half2)) +    // V2
        (16ull * 16 * sizeof(float));     // Sbuf

    sa_forward_v3_kernel<<<grid, block, shmem>>>(
        (const half*)Q.data_ptr<at::Half>(),
        (const half*)K.data_ptr<at::Half>(),
        (const half*)V.data_ptr<at::Half>(),
        (half*)O.data_ptr<at::Half>(),
        B, H, N,
        (float)scale
    );

    return O;
}
