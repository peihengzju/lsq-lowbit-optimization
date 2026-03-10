import torch


def _to_nibble_signed(x: torch.Tensor) -> torch.Tensor:
    # 4-bit two's complement encoding in [0, 15]
    return (x.to(torch.int16) & 0x0F).to(torch.uint8)


def pack_int4_weights(w_int4: torch.Tensor) -> torch.Tensor:
    """
    Pack last dimension int4 values (range [-8, 7]) into uint8.

    Input shape: [..., K], K must be even.
    Output shape: [..., K // 2]
    """
    if w_int4.dtype not in (torch.int8, torch.int16, torch.int32):
        raise TypeError(f"Expected integer tensor, got {w_int4.dtype}")

    if w_int4.shape[-1] % 2 != 0:
        raise ValueError("Last dimension K must be even for int4 packing")

    if torch.any(w_int4 < -8) or torch.any(w_int4 > 7):
        raise ValueError("INT4 tensor contains values outside [-8, 7]")

    lo = _to_nibble_signed(w_int4[..., 0::2])
    hi = _to_nibble_signed(w_int4[..., 1::2])
    return lo | (hi << 4)


def unpack_int4_weights(w_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 tensor where each byte stores two signed int4 values.
    Returns int8 tensor with doubled last dimension.
    """
    if w_packed.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 tensor, got {w_packed.dtype}")

    lo = (w_packed & 0x0F).to(torch.int8)
    hi = ((w_packed >> 4) & 0x0F).to(torch.int8)

    # Sign-extend from 4-bit to 8-bit: (x << 4) >> 4
    lo = (lo << 4) >> 4
    hi = (hi << 4) >> 4

    out_shape = list(w_packed.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.int8, device=w_packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out
