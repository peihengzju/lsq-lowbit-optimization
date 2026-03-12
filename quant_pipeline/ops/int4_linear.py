import torch
import torch.nn as nn

from quant_pipeline.ops.int4_int8_gemm import linear_int4_int8
from quant_pipeline.quantization.int4_pack import pack_int4_weights


class Int4WeightInt8ActLinear(nn.Module):
    """
    Hardware-aware inference linear layer.

    - Weight storage: signed INT4 packed as uint8.
    - Activation compute: INT8.
    - Accumulation: INT32 (from CUDA kernel).
    - Output: FP16/FP32 dequantized tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_scale: float,
        a_scale: float,
        a_qn: int = -127,
        a_qp: int = 127,
        out_dtype: torch.dtype = torch.float16,
        bias: bool = False,
    ):
        super().__init__()
        if in_features % 2 != 0:
            raise ValueError("in_features must be even for INT4 packing")

        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.a_qn = int(a_qn)
        self.a_qp = int(a_qp)
        self.a_scale_host = float(a_scale)
        self.w_scale_host = float(w_scale)

        if self.a_qn < -128 or self.a_qp > 127:
            raise ValueError(
                f"Activation quantization range [{self.a_qn}, {self.a_qp}] does not fit int8 compute"
            )

        self.register_buffer("w_scale", torch.tensor(self.w_scale_host, dtype=torch.float32))
        self.register_buffer("a_scale", torch.tensor(self.a_scale_host, dtype=torch.float32))

        self.register_buffer(
            "weight_packed",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def load_int4_weight(self, w_int4: torch.Tensor) -> None:
        if w_int4.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Expected weight shape {(self.out_features, self.in_features)}, got {tuple(w_int4.shape)}"
            )
        if w_int4.dtype != torch.int8:
            raise TypeError("w_int4 must be int8 with values in [-8, 7]")

        packed = pack_int4_weights(w_int4).to(self.weight_packed.device)
        self.weight_packed.copy_(packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input must be rank-2 [batch, in_features]")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Input in_features mismatch: expected {self.in_features}, got {x.shape[1]}"
            )

        return linear_int4_int8(
            x.contiguous().to(torch.float32),
            self.weight_packed.contiguous(),
            self.a_scale_host,
            self.w_scale_host,
            self.a_qn,
            self.a_qp,
            self.bias,
            self.out_dtype,
        )
