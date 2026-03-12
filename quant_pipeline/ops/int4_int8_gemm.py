import torch

from quant_pipeline.quantization.int4_pack import pack_int4_weights

try:
    import int4_int8_ext
except ImportError as exc:
    int4_int8_ext = None
    _import_error = exc


def _check_ext_loaded() -> None:
    if int4_int8_ext is None:
        raise RuntimeError(
            "CUDA extension int4_int8_ext is not built. "
            "Run: pip install -e ."
        ) from _import_error


def int8_int8_gemm(a_int8: torch.Tensor, b_int8: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B^T using int8 inputs and int32 accumulation.

    A: [M, K] int8 CUDA contiguous
    B: [N, K] int8 CUDA contiguous
    C: [M, N] int32 CUDA contiguous
    """
    _check_ext_loaded()
    return int4_int8_ext.int8_int8_gemm(a_int8, b_int8)


def int4_int8_gemm(a_int8: torch.Tensor, b_int4_or_packed: torch.Tensor) -> torch.Tensor:
    """
    INT4 storage + INT8 compute GEMM.

    A: [M, K] int8
    B input options:
      - [N, K] int8 values in [-8, 7] (will be packed on-the-fly)
      - [N, K//2] uint8 pre-packed
    Returns int32 [M, N]
    """
    _check_ext_loaded()

    if b_int4_or_packed.dtype == torch.uint8:
        b_packed = b_int4_or_packed
    else:
        b_packed = pack_int4_weights(b_int4_or_packed)

    return int4_int8_ext.int4_int8_gemm(a_int8, b_packed)


def lower_quantize_3x3_s1_p1(
    x_fp32: torch.Tensor,
    scale: float,
    qn: int,
    qp: int,
    padded_k: int,
) -> torch.Tensor:
    """
    Fused quantize + lowering for 3x3 stride-1 padding-1 convolutions.

    Input: [N, C, H, W] float32 CUDA contiguous
    Output: [N*H*W, padded_k] int8 CUDA contiguous
    """
    _check_ext_loaded()
    return int4_int8_ext.lower_quantize_3x3_s1_p1(
        x_fp32,
        float(scale),
        int(qn),
        int(qp),
        int(padded_k),
    )


def conv3x3_int4_int8_s1_p1(
    x_fp32: torch.Tensor,
    w_packed: torch.Tensor,
    a_scale: float,
    w_scale: float,
    qn: int,
    qp: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Fused 3x3 stride-1 padding-1 conv path.

    Performs quantize + lowering + INT4xINT8 GEMM + dequant/bias/output layout in CUDA.
    """
    _check_ext_loaded()
    output_half = out_dtype == torch.float16
    return int4_int8_ext.conv3x3_int4_int8_s1_p1(
        x_fp32,
        w_packed,
        float(a_scale),
        float(w_scale),
        int(qn),
        int(qp),
        bias,
        bool(output_half),
    )


def linear_int4_int8(
    x_fp32: torch.Tensor,
    w_packed: torch.Tensor,
    a_scale: float,
    w_scale: float,
    qn: int,
    qp: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    _check_ext_loaded()
    output_half = out_dtype == torch.float16
    return int4_int8_ext.linear_int4_int8(
        x_fp32,
        w_packed,
        float(a_scale),
        float(w_scale),
        int(qn),
        int(qp),
        bias,
        bool(output_half),
    )


def conv1x1_int4_int8(
    x_fp32: torch.Tensor,
    w_packed: torch.Tensor,
    a_scale: float,
    w_scale: float,
    qn: int,
    qp: int,
    stride_h: int,
    stride_w: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    _check_ext_loaded()
    output_half = out_dtype == torch.float16
    return int4_int8_ext.conv1x1_int4_int8(
        x_fp32,
        w_packed,
        float(a_scale),
        float(w_scale),
        int(qn),
        int(qp),
        int(stride_h),
        int(stride_w),
        bias,
        bool(output_half),
    )
