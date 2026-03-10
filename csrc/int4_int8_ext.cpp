#include <torch/extension.h>
#include <vector>

at::Tensor int8_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b);
at::Tensor int4_int8_gemm_cuda(const at::Tensor& a, const at::Tensor& b_packed);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_int8_gemm", &int8_int8_gemm, "INT8xINT8 GEMM (CUDA)");
    m.def("int4_int8_gemm", &int4_int8_gemm, "INT4 storage + INT8 compute GEMM (CUDA)");
}
