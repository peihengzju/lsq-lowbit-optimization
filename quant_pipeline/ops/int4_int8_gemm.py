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
