import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_pipeline.ops.int4_int8_gemm import conv1x1_int4_int8, conv3x3_int4_int8_s1_p1, int4_int8_gemm
from quant_pipeline.quantization.int4_pack import pack_int4_weights


class Int4WeightInt8ActConv2d(nn.Module):
    """
    Conv2d lowered to unfold + INT4xINT8 GEMM.

    This is an integration path for full-model evaluation, not a final optimized kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        w_scale: float,
        a_scale: float,
        a_qn: int,
        a_qp: int,
        out_dtype: torch.dtype = torch.float16,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise ValueError("Only groups=1 is supported")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(int(v) for v in kernel_size)
        self.stride = tuple(int(v) for v in stride)
        self.padding = tuple(int(v) for v in padding)
        self.dilation = tuple(int(v) for v in dilation)
        self.groups = int(groups)
        self.out_dtype = out_dtype
        self.a_qn = int(a_qn)
        self.a_qp = int(a_qp)
        self.a_scale_host = float(a_scale)
        self.w_scale_host = float(w_scale)

        if self.a_qn < -128 or self.a_qp > 127:
            raise ValueError(
                f"Activation quantization range [{self.a_qn}, {self.a_qp}] does not fit int8 compute"
            )

        self.flat_in_features = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.padded_in_features = self.flat_in_features + (self.flat_in_features % 2)
        self.is_pointwise = (
            self.kernel_size == (1, 1)
            and self.stride == (1, 1)
            and self.padding == (0, 0)
            and self.dilation == (1, 1)
        )
        self.is_spatial_3x3 = (
            self.kernel_size == (3, 3)
            and self.stride == (1, 1)
            and self.padding == (1, 1)
            and self.dilation == (1, 1)
        )

        self.register_buffer("w_scale", torch.tensor(self.w_scale_host, dtype=torch.float32))
        self.register_buffer("a_scale", torch.tensor(self.a_scale_host, dtype=torch.float32))
        self.register_buffer(
            "weight_packed",
            torch.zeros((self.out_channels, self.padded_in_features // 2), dtype=torch.uint8),
        )
        self._cols_buffer: torch.Tensor | None = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def load_int4_weight(self, w_int4: torch.Tensor) -> None:
        expected_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if tuple(w_int4.shape) != expected_shape:
            raise ValueError(f"Expected weight shape {expected_shape}, got {tuple(w_int4.shape)}")
        if w_int4.dtype != torch.int8:
            raise TypeError("w_int4 must be int8 with values in [-8, 7]")

        w_flat = w_int4.reshape(self.out_channels, -1)
        if self.padded_in_features != self.flat_in_features:
            padded = torch.zeros(
                (self.out_channels, self.padded_in_features),
                dtype=torch.int8,
                device=w_flat.device,
            )
            padded[:, : self.flat_in_features] = w_flat
            w_flat = padded

        packed = pack_int4_weights(w_flat).to(self.weight_packed.device)
        self.weight_packed.copy_(packed)

    def _output_hw(self, h: int, w: int) -> Tuple[int, int]:
        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            / self.stride[0]
            + 1
        )
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            / self.stride[1]
            + 1
        )
        return h_out, w_out

    def _get_cols_buffer(self, rows: int, device: torch.device) -> torch.Tensor:
        if (
            self._cols_buffer is None
            or self._cols_buffer.device != device
            or self._cols_buffer.shape != (rows, self.padded_in_features)
        ):
            self._cols_buffer = torch.empty(
                (rows, self.padded_in_features),
                dtype=torch.int8,
                device=device,
            )
        return self._cols_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input must be rank-4 [batch, channels, height, width]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )

        if self.is_spatial_3x3:
            return conv3x3_int4_int8_s1_p1(
                x.contiguous().to(torch.float32),
                self.weight_packed.contiguous(),
                self.a_scale_host,
                self.w_scale_host,
                self.a_qn,
                self.a_qp,
                self.bias,
                self.out_dtype,
            )

        if self.is_pointwise:
            return conv1x1_int4_int8(
                x.contiguous().to(torch.float32),
                self.weight_packed.contiguous(),
                self.a_scale_host,
                self.w_scale_host,
                self.a_qn,
                self.a_qp,
                self.stride[0],
                self.stride[1],
                self.bias,
                self.out_dtype,
            )

        # Quantize once in float, then cast to int8 after lowering.
        x_quant = torch.clamp(torch.round(x / self.a_scale), self.a_qn, self.a_qp)

        x_cols_src = F.unfold(
            x_quant,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        x_cols_src = x_cols_src.transpose(1, 2).reshape(-1, self.flat_in_features)

        # Reuse a cached int8 lowering buffer to reduce per-forward allocations.
        x_cols = self._get_cols_buffer(x_cols_src.shape[0], x.device)
        x_cols_view = x_cols[:, : self.flat_in_features]
        x_cols_view.copy_(x_cols_src)
        if self.padded_in_features != self.flat_in_features:
            x_cols[:, self.flat_in_features :].zero_()

        acc_int32 = int4_int8_gemm(x_cols.contiguous(), self.weight_packed.contiguous())
        y = acc_int32.to(torch.float32) * (self.a_scale * self.w_scale)

        if self.bias is not None:
            y = y + self.bias

        n, _, h, w = x.shape
        h_out, w_out = self._output_hw(h, w)
        y = y.view(n, h_out * w_out, self.out_channels).transpose(1, 2).reshape(
            n, self.out_channels, h_out, w_out
        )
        return y.to(self.out_dtype)
