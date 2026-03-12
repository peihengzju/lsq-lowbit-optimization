#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_61_intrinsics.h>
#include <cuda_fp16.h>
#include <type_traits>

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

template <typename scalar_t>
__global__ void postprocess_conv3x3_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ bias,
    scalar_t* __restrict__ y,
    int n,
    int out_channels,
    int h,
    int w,
    float combined_scale) {

    const int total = n * out_channels * h * w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int hw = h * w;
    const int n_idx = idx / (out_channels * hw);
    const int oc_idx = (idx / hw) % out_channels;
    const int hw_idx = idx % hw;
    const int row_idx = n_idx * hw + hw_idx;

    float val = static_cast<float>(acc[row_idx * out_channels + oc_idx]) * combined_scale;
    if (bias != nullptr) {
        val += bias[oc_idx];
    }

    if constexpr (std::is_same_v<scalar_t, at::Half>) {
        y[idx] = __float2half_rn(val);
    } else {
        y[idx] = val;
    }
}

template <typename scalar_t>
__global__ void postprocess_linear_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ bias,
    scalar_t* __restrict__ y,
    int m,
    int out_features,
    float combined_scale) {

    const int total = m * out_features;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int of_idx = idx % out_features;
    float val = static_cast<float>(acc[idx]) * combined_scale;
    if (bias != nullptr) {
        val += bias[of_idx];
    }

    if constexpr (std::is_same_v<scalar_t, at::Half>) {
        y[idx] = __float2half_rn(val);
    } else {
        y[idx] = val;
    }
}

__global__ void quantize_2d_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    int m,
    int k,
    float scale,
    int qn,
    int qp) {
    const int total = m * k;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int q = __float2int_rn(x[idx] / scale);
    q = max(qn, min(qp, q));
    x_q[idx] = static_cast<int8_t>(q);
}

__global__ void lower_quantize_1x1_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ cols,
    int n,
    int c,
    int h,
    int w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    float scale,
    int qn,
    int qp) {
    const int total = n * h_out * w_out * c;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int row_idx = idx / c;
    const int c_idx = idx % c;
    const int n_idx = row_idx / (h_out * w_out);
    const int hw_idx = row_idx % (h_out * w_out);
    const int out_h = hw_idx / w_out;
    const int out_w = hw_idx % w_out;
    const int in_h = out_h * stride_h;
    const int in_w = out_w * stride_w;
    const int offset = ((n_idx * c + c_idx) * h + in_h) * w + in_w;

    int q = __float2int_rn(x[offset] / scale);
    q = max(qn, min(qp, q));
    cols[idx] = static_cast<int8_t>(q);
}

__global__ void lower_quantize_3x3_s1_p1_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ cols,
    int n,
    int c,
    int h,
    int w,
    int padded_k,
    float scale,
    int qn,
    int qp) {

    const int k_per_row = c * 9;
    const int total = n * h * w * padded_k;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int row_idx = idx / padded_k;
    const int col_idx = idx % padded_k;

    if (col_idx >= k_per_row) {
        cols[idx] = 0;
        return;
    }

    const int n_idx = row_idx / (h * w);
    const int hw_idx = row_idx % (h * w);
    const int out_h = hw_idx / w;
    const int out_w = hw_idx % w;

    const int c_idx = col_idx / 9;
    const int k_idx = col_idx % 9;
    const int ky = k_idx / 3;
    const int kx = k_idx % 3;

    const int in_h = out_h + ky - 1;
    const int in_w = out_w + kx - 1;

    float val = 0.0f;
    if (in_h >= 0 && in_h < h && in_w >= 0 && in_w < w) {
        const int offset = ((n_idx * c + c_idx) * h + in_h) * w + in_w;
        val = x[offset];
    }

    int q = __float2int_rn(val / scale);
    q = max(qn, min(qp, q));
    cols[idx] = static_cast<int8_t>(q);
}

}  // namespace

namespace {

template <bool PACKED_INT4_B>
at::Tensor launch_gemm_cuda(
    const at::Tensor& a,
    const at::Tensor& b_int8,
    const at::Tensor& b_packed) {
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = PACKED_INT4_B ? static_cast<int>(b_packed.size(0)) : static_cast<int>(b_int8.size(0));

    auto c = at::zeros({m, n}, a.options().dtype(at::kInt));

    dim3 block(BN, BM);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    gemm_kernel<PACKED_INT4_B><<<grid, block>>>(
        a.data_ptr<int8_t>(),
        PACKED_INT4_B ? nullptr : b_int8.data_ptr<int8_t>(),
        PACKED_INT4_B ? b_packed.data_ptr<uint8_t>() : nullptr,
        c.data_ptr<int32_t>(),
        m,
        n,
        k);

    return c;
}

}  // namespace

at::Tensor int8_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b) {
    return launch_gemm_cuda<false>(a, b, at::Tensor());
}

at::Tensor int4_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b_packed) {
    return launch_gemm_cuda<true>(a, at::Tensor(), b_packed);
}

at::Tensor lower_quantize_3x3_s1_p1_cuda(const at::Tensor& x, double scale, int64_t qn, int64_t qp, int64_t padded_k) {
    const int n = static_cast<int>(x.size(0));
    const int c = static_cast<int>(x.size(1));
    const int h = static_cast<int>(x.size(2));
    const int w = static_cast<int>(x.size(3));
    const int k_per_row = c * 9;
    const int rows = n * h * w;
    const int padded_k_i = static_cast<int>(padded_k);

    auto cols = at::empty({rows, padded_k_i}, x.options().dtype(at::kChar));

    const int total = rows * padded_k_i;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    lower_quantize_3x3_s1_p1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        cols.data_ptr<int8_t>(),
        n,
        c,
        h,
        w,
        padded_k_i,
        static_cast<float>(scale),
        static_cast<int>(qn),
        static_cast<int>(qp));

    return cols;
}

at::Tensor conv3x3_int4_int8_s1_p1_cuda(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    const c10::optional<at::Tensor>& bias,
    bool output_half) {
    const int n = static_cast<int>(x.size(0));
    const int c = static_cast<int>(x.size(1));
    const int h = static_cast<int>(x.size(2));
    const int w = static_cast<int>(x.size(3));
    const int out_channels = static_cast<int>(w_packed.size(0));
    const int padded_k = static_cast<int>(w_packed.size(1) * 2);

    auto cols = lower_quantize_3x3_s1_p1_cuda(x, a_scale, qn, qp, padded_k);
    auto acc = launch_gemm_cuda<true>(cols, at::Tensor(), w_packed);

    const int total = n * out_channels * h * w;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const float combined_scale = static_cast<float>(a_scale * w_scale);
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    if (output_half) {
        auto y = at::empty({n, out_channels, h, w}, x.options().dtype(at::kHalf));
        postprocess_conv3x3_kernel<at::Half><<<blocks, threads>>>(
            acc.data_ptr<int32_t>(),
            bias_ptr,
            y.data_ptr<at::Half>(),
            n,
            out_channels,
            h,
            w,
            combined_scale);
        return y;
    }

    auto y = at::empty({n, out_channels, h, w}, x.options().dtype(at::kFloat));
    postprocess_conv3x3_kernel<float><<<blocks, threads>>>(
        acc.data_ptr<int32_t>(),
        bias_ptr,
        y.data_ptr<float>(),
        n,
        out_channels,
        h,
        w,
        combined_scale);
    return y;
}

at::Tensor linear_int4_int8_cuda(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    const c10::optional<at::Tensor>& bias,
    bool output_half) {
    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int out_features = static_cast<int>(w_packed.size(0));

    auto x_q = at::empty({m, k}, x.options().dtype(at::kChar));
    constexpr int threads = 256;
    int blocks = (m * k + threads - 1) / threads;
    quantize_2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        x_q.data_ptr<int8_t>(),
        m,
        k,
        static_cast<float>(a_scale),
        static_cast<int>(qn),
        static_cast<int>(qp));

    auto acc = launch_gemm_cuda<true>(x_q, at::Tensor(), w_packed);
    const int total = m * out_features;
    blocks = (total + threads - 1) / threads;
    const float combined_scale = static_cast<float>(a_scale * w_scale);
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    if (output_half) {
        auto y = at::empty({m, out_features}, x.options().dtype(at::kHalf));
        postprocess_linear_kernel<at::Half><<<blocks, threads>>>(
            acc.data_ptr<int32_t>(),
            bias_ptr,
            y.data_ptr<at::Half>(),
            m,
            out_features,
            combined_scale);
        return y;
    }

    auto y = at::empty({m, out_features}, x.options().dtype(at::kFloat));
    postprocess_linear_kernel<float><<<blocks, threads>>>(
        acc.data_ptr<int32_t>(),
        bias_ptr,
        y.data_ptr<float>(),
        m,
        out_features,
        combined_scale);
    return y;
}

at::Tensor conv1x1_int4_int8_cuda(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    int64_t stride_h,
    int64_t stride_w,
    const c10::optional<at::Tensor>& bias,
    bool output_half) {
    const int n = static_cast<int>(x.size(0));
    const int c = static_cast<int>(x.size(1));
    const int h = static_cast<int>(x.size(2));
    const int w = static_cast<int>(x.size(3));
    const int out_channels = static_cast<int>(w_packed.size(0));
    const int sh = static_cast<int>(stride_h);
    const int sw = static_cast<int>(stride_w);
    const int h_out = (h - 1) / sh + 1;
    const int w_out = (w - 1) / sw + 1;
    const int rows = n * h_out * w_out;

    auto cols = at::empty({rows, c}, x.options().dtype(at::kChar));
    constexpr int threads = 256;
    int blocks = (rows * c + threads - 1) / threads;
    lower_quantize_1x1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        cols.data_ptr<int8_t>(),
        n,
        c,
        h,
        w,
        h_out,
        w_out,
        sh,
        sw,
        static_cast<float>(a_scale),
        static_cast<int>(qn),
        static_cast<int>(qp));

    auto acc = launch_gemm_cuda<true>(cols, at::Tensor(), w_packed);
    const int total = n * out_channels * h_out * w_out;
    blocks = (total + threads - 1) / threads;
    const float combined_scale = static_cast<float>(a_scale * w_scale);
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    if (output_half) {
        auto y = at::empty({n, out_channels, h_out, w_out}, x.options().dtype(at::kHalf));
        postprocess_conv3x3_kernel<at::Half><<<blocks, threads>>>(
            acc.data_ptr<int32_t>(),
            bias_ptr,
            y.data_ptr<at::Half>(),
            n,
            out_channels,
            h_out,
            w_out,
            combined_scale);
        return y;
    }

    auto y = at::empty({n, out_channels, h_out, w_out}, x.options().dtype(at::kFloat));
    postprocess_conv3x3_kernel<float><<<blocks, threads>>>(
        acc.data_ptr<int32_t>(),
        bias_ptr,
        y.data_ptr<float>(),
        n,
        out_channels,
        h_out,
        w_out,
        combined_scale);
    return y;
}
