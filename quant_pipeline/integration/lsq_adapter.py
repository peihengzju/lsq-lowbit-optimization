from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from quant_pipeline.ops import Int4WeightInt8ActConv2d, Int4WeightInt8ActLinear


@dataclass
class ConversionResult:
    name: str
    module_type: str
    converted: bool
    reason: str


def _resolve_num_classes(state: dict[str, torch.Tensor], default: int = 1000) -> int:
    if "fc.weight" in state:
        return int(state["fc.weight"].shape[0])
    if "fc.linear.weight" in state:
        return int(state["fc.linear.weight"].shape[0])
    return default


def _is_lsq_checkpoint(state: dict[str, torch.Tensor]) -> bool:
    return any(
        ".w_quant." in k or ".a_quant." in k or k.endswith(".conv.weight")
        for k in state.keys()
    )


def build_lsq_model_from_ckpt(
    lsq_root: str | Path,
    ckpt_path: str | Path,
    w_bits: int = 4,
    a_bits: int = 4,
    first_last_bits: int = 8,
    disable_first_last_8bit: bool = False,
    signed_input_first_layer: bool = False,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Build LSQ model from external repository and load checkpoint.

    This function imports from the user LSQ repo at runtime.
    """
    import sys

    lsq_root = Path(lsq_root).resolve()
    if not lsq_root.exists():
        raise FileNotFoundError(f"LSQ root not found: {lsq_root}")

    if str(lsq_root) not in sys.path:
        sys.path.insert(0, str(lsq_root))

    from lsq.models import LSQConfig, apply_lsq_quantization, preact_resnet18

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    is_lsq = _is_lsq_checkpoint(state)

    if num_classes is None:
        num_classes = _resolve_num_classes(state)

    model = preact_resnet18(num_classes=num_classes)
    if is_lsq:
        cfg = LSQConfig(
            w_bits=w_bits,
            a_bits=a_bits,
            first_last_bits=first_last_bits,
            quantize_first_last_8bit=not disable_first_last_8bit,
            signed_input_first_layer=signed_input_first_layer,
        )
        apply_lsq_quantization(model, cfg)
    model.load_state_dict(state, strict=True)
    return model


def _is_lsq_quant_module(module: nn.Module) -> bool:
    return hasattr(module, "w_quant") and hasattr(module, "a_quant") and (
        hasattr(module, "linear") or hasattr(module, "conv")
    )


def _replace_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def _get_quant_range(quantizer: nn.Module) -> tuple[int, int]:
    qn = int(getattr(quantizer, "qn"))
    qp = int(getattr(quantizer, "qp"))
    return qn, qp


def _quantize_tensor(v: torch.Tensor, scale: float, qn: int, qp: int) -> torch.Tensor:
    return torch.clamp(torch.round(v / scale), qn, qp).to(torch.int8)


def _convert_quant_linear_module(name: str, module: nn.Module) -> tuple[Optional[nn.Module], str]:
    if not isinstance(module.linear, nn.Linear):
        return None, "linear child is not nn.Linear"

    w_quant = module.w_quant
    a_quant = module.a_quant
    linear = module.linear
    w_qn, w_qp = _get_quant_range(w_quant)
    a_qn, a_qp = _get_quant_range(a_quant)

    if (w_qn, w_qp) != (-8, 7):
        return None, f"weight range [{w_qn}, {w_qp}] is not signed INT4"
    if a_qn < -128 or a_qp > 127:
        return None, f"activation range [{a_qn}, {a_qp}] does not fit int8 compute"
    if linear.in_features % 2 != 0:
        return None, "in_features is odd and current int4 linear path cannot pack it"

    w_scale = float(torch.clamp(w_quant.s.detach(), min=1e-8).item())
    a_scale = float(torch.clamp(a_quant.s.detach(), min=1e-8).item())

    fused = Int4WeightInt8ActLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        w_scale=w_scale,
        a_scale=a_scale,
        a_qn=a_qn,
        a_qp=a_qp,
        out_dtype=torch.float32,
        bias=linear.bias is not None,
    ).to(linear.weight.device)

    w_int4 = _quantize_tensor(linear.weight.detach(), w_scale, w_qn, w_qp)
    fused.load_int4_weight(w_int4)

    if linear.bias is not None:
        fused.bias.data.copy_(linear.bias.detach().to(fused.bias.dtype))

    return fused, "converted"


def _convert_quant_conv_module(name: str, module: nn.Module) -> tuple[Optional[nn.Module], str]:
    if not isinstance(module.conv, nn.Conv2d):
        return None, "conv child is not nn.Conv2d"

    w_quant = module.w_quant
    a_quant = module.a_quant
    conv = module.conv
    w_qn, w_qp = _get_quant_range(w_quant)
    a_qn, a_qp = _get_quant_range(a_quant)

    if (w_qn, w_qp) != (-8, 7):
        return None, f"weight range [{w_qn}, {w_qp}] is not signed INT4"
    if a_qn < -128 or a_qp > 127:
        return None, f"activation range [{a_qn}, {a_qp}] does not fit int8 compute"
    if conv.groups != 1:
        return None, "grouped convolution is not supported"

    w_scale = float(torch.clamp(w_quant.s.detach(), min=1e-8).item())
    a_scale = float(torch.clamp(a_quant.s.detach(), min=1e-8).item())

    fused = Int4WeightInt8ActConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        w_scale=w_scale,
        a_scale=a_scale,
        a_qn=a_qn,
        a_qp=a_qp,
        out_dtype=torch.float32,
        bias=conv.bias is not None,
    ).to(conv.weight.device)

    w_int4 = _quantize_tensor(conv.weight.detach(), w_scale, w_qn, w_qp)
    fused.load_int4_weight(w_int4)

    if conv.bias is not None:
        fused.bias.data.copy_(conv.bias.detach().to(fused.bias.dtype))

    return fused, "converted"


@torch.no_grad()
def convert_lsq_modules(
    model: nn.Module,
    only_names: Optional[List[str]] = None,
    convert_linear: bool = True,
    convert_conv: bool = False,
) -> List[ConversionResult]:
    results: List[ConversionResult] = []

    for name, module in list(model.named_modules()):
        if only_names is not None and name not in only_names:
            continue
        if not _is_lsq_quant_module(module):
            continue

        converted_module: Optional[nn.Module] = None
        reason = "unsupported module"
        module_type = type(module).__name__

        if hasattr(module, "linear"):
            module_type = "QuantLinear"
            if convert_linear:
                converted_module, reason = _convert_quant_linear_module(name, module)
            else:
                reason = "linear conversion disabled"
        elif hasattr(module, "conv"):
            module_type = "QuantConv2d"
            if convert_conv:
                converted_module, reason = _convert_quant_conv_module(name, module)
            else:
                reason = "conv conversion disabled"

        if converted_module is not None:
            _replace_module(model, name, converted_module)
            results.append(
                ConversionResult(name=name, module_type=module_type, converted=True, reason=reason)
            )
        else:
            results.append(
                ConversionResult(name=name, module_type=module_type, converted=False, reason=reason)
            )

    return results


@torch.no_grad()
def convert_quant_linear_modules(model: nn.Module, only_names: Optional[List[str]] = None) -> List[str]:
    converted = convert_lsq_modules(
        model,
        only_names=only_names,
        convert_linear=True,
        convert_conv=False,
    )
    return [item.name for item in converted if item.converted]
