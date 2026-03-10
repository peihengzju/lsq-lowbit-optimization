#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_61_intrinsics.h>

namespace {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 32;
static_assert(BK % 4 == 0, "BK must be divisible by 4 for dp4a");

__device__ __forceinline__ int8_t sign_extend_int4(uint8_t x) {
    return static_cast<int8_t>((x & 0x0F) | ((x & 0x08) ? 0xF0 : 0x00));
}

template <bool PACKED_INT4_B>
__global__ void gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b_int8,
    const uint8_t* __restrict__ b_packed,
    int32_t* __restrict__ c,
    int m,
    int n,
    int k) {

    __shared__ int8_t a_tile[BM][BK];
    __shared__ int8_t b_tile[BN][BK];

    const int row = blockIdx.y * BM + threadIdx.y;
    const int col = blockIdx.x * BN + threadIdx.x;

    int32_t acc = 0;

    for (int k0 = 0; k0 < k; k0 += BK) {
        // Load A tile
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < BM * BK; idx += blockDim.x * blockDim.y) {
            const int r = idx / BK;
            const int kk = idx % BK;
            const int gr = blockIdx.y * BM + r;
            const int gk = k0 + kk;
            a_tile[r][kk] = (gr < m && gk < k) ? a[gr * k + gk] : 0;
        }

        // Load B tile (with optional int4 unpack)
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < BN * BK; idx += blockDim.x * blockDim.y) {
            const int c_local = idx / BK;
            const int kk = idx % BK;
            const int gc = blockIdx.x * BN + c_local;
            const int gk = k0 + kk;

            int8_t val = 0;
            if (gc < n && gk < k) {
                if constexpr (PACKED_INT4_B) {
                    const uint8_t packed = b_packed[gc * (k / 2) + (gk >> 1)];
                    const uint8_t nibble = (gk & 1) ? (packed >> 4) : (packed & 0x0F);
                    val = sign_extend_int4(nibble);
                } else {
                    val = b_int8[gc * k + gk];
                }
            }
            b_tile[c_local][kk] = val;
        }

        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll
            for (int kk = 0; kk < BK; kk += 4) {
                const int a_vec = *reinterpret_cast<const int*>(&a_tile[threadIdx.y][kk]);
                const int b_vec = *reinterpret_cast<const int*>(&b_tile[threadIdx.x][kk]);
                acc = __dp4a(a_vec, b_vec, acc);
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

}  // namespace

at::Tensor int8_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b) {
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(0));

    auto c = at::zeros({m, n}, a.options().dtype(at::kInt));

    dim3 block(BN, BM);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    gemm_kernel<false><<<grid, block>>>(
        a.data_ptr<int8_t>(),
        b.data_ptr<int8_t>(),
        nullptr,
        c.data_ptr<int32_t>(),
        m,
        n,
        k);

    return c;
}

at::Tensor int4_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b_packed) {
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b_packed.size(0));

    auto c = at::zeros({m, n}, a.options().dtype(at::kInt));

    dim3 block(BN, BM);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    gemm_kernel<true><<<grid, block>>>(
        a.data_ptr<int8_t>(),
        nullptr,
        b_packed.data_ptr<uint8_t>(),
        c.data_ptr<int32_t>(),
        m,
        n,
        k);

    return c;
}
