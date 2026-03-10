import torch

from quant_pipeline.quantization import pack_int4_weights, unpack_int4_weights


def test_pack_unpack_roundtrip_cpu():
    x = torch.randint(-8, 8, (32, 128), dtype=torch.int8)
    packed = pack_int4_weights(x)
    unpacked = unpack_int4_weights(packed)
    assert torch.equal(x, unpacked)


def test_pack_unpack_roundtrip_cuda_if_available():
    if not torch.cuda.is_available():
        return
    x = torch.randint(-8, 8, (64, 256), dtype=torch.int8, device="cuda")
    packed = pack_int4_weights(x)
    unpacked = unpack_int4_weights(packed)
    assert torch.equal(x, unpacked)
