#include <torch/extension.h>
#include <vector>

at::Tensor int8_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b);
at::Tensor int4_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b_packed);
at::Tensor lower_quantize_3x3_s1_p1_cuda(const at::Tensor& x, double scale, int64_t qn, int64_t qp, int64_t padded_k);
at::Tensor conv3x3_int4_int8_s1_p1_cuda(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    const c10::optional<at::Tensor>& bias,
    bool output_half);
at::Tensor linear_int4_int8_cuda(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    const c10::optional<at::Tensor>& bias,
    bool output_half);
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
    bool output_half);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor int8_int8_gemm(const at::Tensor& a, const at::Tensor& b) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    CHECK_CONTIGUOUS(a);
    CHECK_CONTIGUOUS(b);

    TORCH_CHECK(a.dim() == 2, "A must be rank-2 [M, K]");
    TORCH_CHECK(b.dim() == 2, "B must be rank-2 [N, K]");
    TORCH_CHECK(a.scalar_type() == at::kChar, "A must be int8");
    TORCH_CHECK(b.scalar_type() == at::kChar, "B must be int8");
    TORCH_CHECK(a.size(1) == b.size(1), "K dimension mismatch");

    return int8_int8_gemm_cuda(a, b);
}

at::Tensor int4_int8_gemm(const at::Tensor& a, const at::Tensor& b_packed) {
    CHECK_CUDA(a);
    CHECK_CUDA(b_packed);
    CHECK_CONTIGUOUS(a);
    CHECK_CONTIGUOUS(b_packed);

    TORCH_CHECK(a.dim() == 2, "A must be rank-2 [M, K]");
    TORCH_CHECK(b_packed.dim() == 2, "B packed must be rank-2 [N, K/2]");
    TORCH_CHECK(a.scalar_type() == at::kChar, "A must be int8");
    TORCH_CHECK(b_packed.scalar_type() == at::kByte, "B packed must be uint8");

    const auto k = a.size(1);
    TORCH_CHECK(k % 2 == 0, "K must be even for int4 packed weights");
    TORCH_CHECK(b_packed.size(1) * 2 == k, "Packed B last dim must be K/2");

    return int4_int8_gemm_cuda(a, b_packed);
}

at::Tensor lower_quantize_3x3_s1_p1(const at::Tensor& x, double scale, int64_t qn, int64_t qp, int64_t padded_k) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);

    TORCH_CHECK(x.dim() == 4, "x must be rank-4 [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(qn >= -128 && qp <= 127, "quant range must fit int8");
    TORCH_CHECK(scale > 0.0, "scale must be positive");
    TORCH_CHECK(padded_k >= x.size(1) * 9, "padded_k must cover C*3*3");

    return lower_quantize_3x3_s1_p1_cuda(x, scale, qn, qp, padded_k);
}

at::Tensor conv3x3_int4_int8_s1_p1(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    const c10::optional<at::Tensor>& bias,
    bool output_half) {
    CHECK_CUDA(x);
    CHECK_CUDA(w_packed);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w_packed);

    TORCH_CHECK(x.dim() == 4, "x must be rank-4 [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be rank-2 [O, padded_k/2]");
    TORCH_CHECK(w_packed.scalar_type() == at::kByte, "w_packed must be uint8");
    TORCH_CHECK(a_scale > 0.0 && w_scale > 0.0, "scales must be positive");
    TORCH_CHECK(qn >= -128 && qp <= 127, "quant range must fit int8");

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be rank-1 [O]");
        TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias->size(0) == w_packed.size(0), "bias size must match out_channels");
    }

    return conv3x3_int4_int8_s1_p1_cuda(
        x,
        w_packed,
        a_scale,
        w_scale,
        qn,
        qp,
        bias,
        output_half);
}

at::Tensor linear_int4_int8(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double a_scale,
    double w_scale,
    int64_t qn,
    int64_t qp,
    const c10::optional<at::Tensor>& bias,
    bool output_half) {
    CHECK_CUDA(x);
    CHECK_CUDA(w_packed);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w_packed);

    TORCH_CHECK(x.dim() == 2, "x must be rank-2 [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be rank-2 [O, K/2]");
    TORCH_CHECK(w_packed.scalar_type() == at::kByte, "w_packed must be uint8");
    TORCH_CHECK(a_scale > 0.0 && w_scale > 0.0, "scales must be positive");
    TORCH_CHECK(qn >= -128 && qp <= 127, "quant range must fit int8");
    TORCH_CHECK(x.size(1) == w_packed.size(1) * 2, "input K must match packed weight K");

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be rank-1 [O]");
        TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias->size(0) == w_packed.size(0), "bias size must match out_features");
    }

    return linear_int4_int8_cuda(x, w_packed, a_scale, w_scale, qn, qp, bias, output_half);
}

at::Tensor conv1x1_int4_int8(
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
    CHECK_CUDA(x);
    CHECK_CUDA(w_packed);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w_packed);

    TORCH_CHECK(x.dim() == 4, "x must be rank-4 [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be rank-2 [O, C/2]");
    TORCH_CHECK(w_packed.scalar_type() == at::kByte, "w_packed must be uint8");
    TORCH_CHECK(a_scale > 0.0 && w_scale > 0.0, "scales must be positive");
    TORCH_CHECK(qn >= -128 && qp <= 127, "quant range must fit int8");
    TORCH_CHECK(x.size(1) == w_packed.size(1) * 2, "input C must match packed weight K");
    TORCH_CHECK(stride_h >= 1 && stride_w >= 1, "stride must be positive");

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be rank-1 [O]");
        TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias->size(0) == w_packed.size(0), "bias size must match out_channels");
    }

    return conv1x1_int4_int8_cuda(
        x,
        w_packed,
        a_scale,
        w_scale,
        qn,
        qp,
        stride_h,
        stride_w,
        bias,
        output_half);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_int8_gemm", &int8_int8_gemm, "INT8xINT8 GEMM (CUDA)");
    m.def("int4_int8_gemm", &int4_int8_gemm, "INT4 storage + INT8 compute GEMM (CUDA)");
    m.def("lower_quantize_3x3_s1_p1", &lower_quantize_3x3_s1_p1, "Fused quantize+lower for 3x3 s1 p1 conv (CUDA)");
    m.def("conv3x3_int4_int8_s1_p1", &conv3x3_int4_int8_s1_p1, "Fused 3x3 s1 p1 conv: quantize+lower+gemm+dequant (CUDA)");
    m.def("linear_int4_int8", &linear_int4_int8, "Fused linear: quantize+gemm+dequant (CUDA)");
    m.def("conv1x1_int4_int8", &conv1x1_int4_int8, "Fused 1x1 conv: quantize+lower+gemm+dequant (CUDA)");
}
