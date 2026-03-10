from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from quant_pipeline.ops.int4_linear import Int4WeightInt8ActLinear


def _resolve_num_classes(state: dict[str, torch.Tensor], default: int = 1000) -> int:
    if "fc.weight" in state:
        return int(state["fc.weight"].shape[0])
    if "fc.linear.weight" in state:
        return int(state["fc.linear.weight"].shape[0])
    return default


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

    if num_classes is None:
        num_classes = _resolve_num_classes(state)

    model = preact_resnet18(num_classes=num_classes)
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


@torch.no_grad()
def convert_quant_linear_modules(model: nn.Module, only_names: Optional[List[str]] = None) -> List[str]:
    """
    Replace LSQ QuantLinear modules with INT4-storage INT8-compute modules.

    Note:
    - Only 4-bit signed weight modules are converted.
    - Convolution layers are untouched.
    """
    converted: List[str] = []

    for name, module in list(model.named_modules()):
        if only_names is not None and name not in only_names:
            continue

        # Duck-typing for LSQ QuantLinear from external repo
        if not (hasattr(module, "linear") and hasattr(module, "w_quant") and hasattr(module, "a_quant")):
            continue
        if not isinstance(module.linear, nn.Linear):
            continue

        w_quant = module.w_quant
        a_quant = module.a_quant
        linear = module.linear

        qn = getattr(w_quant, "qn", None)
        qp = getattr(w_quant, "qp", None)
        if qn != -8 or qp != 7:
            # Only convert int4 signed weight modules.
            continue

        w_scale = float(torch.clamp(w_quant.s.detach(), min=1e-8).item())
        a_scale = float(torch.clamp(a_quant.s.detach(), min=1e-8).item())

        fused = Int4WeightInt8ActLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            w_scale=w_scale,
            a_scale=a_scale,
            out_dtype=torch.float16,
            bias=linear.bias is not None,
        )
        fused = fused.to(linear.weight.device)

        w_int4 = torch.clamp(torch.round(linear.weight.detach() / w_scale), -8, 7).to(torch.int8)
        fused.load_int4_weight(w_int4)

        if linear.bias is not None:
            fused.bias.data.copy_(linear.bias.detach().to(fused.bias.dtype))

        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], fused)

        converted.append(name)

    return converted
