import argparse
import time

import torch

from quant_pipeline.integration import build_lsq_model_from_ckpt, convert_quant_linear_modules


def benchmark(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def get_module_by_name(root: torch.nn.Module, name: str) -> torch.nn.Module:
    m = root
    for part in name.split("."):
        m = getattr(m, part)
    return m


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark LSQ QuantLinear vs INT4-storage INT8-compute")
    parser.add_argument(
        "--lsq-root",
        type=str,
        required=True,
        help="Path to the external LSQ repository root.",
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--module-name", type=str, default="fc")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--first-last-bits", type=int, default=8)
    parser.add_argument("--disable-first-last-8bit", action="store_true")
    parser.add_argument("--signed-input-first-layer", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)

    model = build_lsq_model_from_ckpt(
        args.lsq_root,
        args.ckpt,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        disable_first_last_8bit=args.disable_first_last_8bit,
        signed_input_first_layer=args.signed_input_first_layer,
    )
    model = model.cuda().eval()

    qlinear = get_module_by_name(model, args.module_name)
    if not (hasattr(qlinear, "linear") and hasattr(qlinear, "w_quant") and hasattr(qlinear, "a_quant")):
        raise ValueError(f"Module '{args.module_name}' is not an LSQ QuantLinear")

    in_features = qlinear.linear.in_features
    x_fp32 = torch.randn(args.batch_size, in_features, device="cuda", dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)

    with torch.no_grad():
        y_ref = qlinear(x_fp32)

    def run_lsq_quantlinear():
        return qlinear(x_fp32)

    fp16_linear = torch.nn.Linear(in_features, qlinear.linear.out_features, bias=qlinear.linear.bias is not None).cuda().half()
    fp16_linear.weight.data.copy_(qlinear.linear.weight.detach().half())
    if qlinear.linear.bias is not None:
        fp16_linear.bias.data.copy_(qlinear.linear.bias.detach().half())

    def run_fp16_linear():
        return fp16_linear(x_fp16)

    model_int4 = build_lsq_model_from_ckpt(
        args.lsq_root,
        args.ckpt,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        disable_first_last_8bit=args.disable_first_last_8bit,
        signed_input_first_layer=args.signed_input_first_layer,
    ).cuda().eval()
    converted = convert_quant_linear_modules(model_int4, only_names=[args.module_name])
    if not converted:
        raise RuntimeError(
            f"Module '{args.module_name}' was not converted. "
            "Likely this layer is not 4-bit signed (for example fc kept at 8-bit when first/last 8-bit is enabled). "
            "Try --disable-first-last-8bit for fc."
        )
    int4_linear = get_module_by_name(model_int4, args.module_name)

    def run_int4_storage():
        return int4_linear(x_fp32)

    with torch.no_grad():
        y_int4 = run_int4_storage()
        max_abs_err = (y_ref - y_int4).abs().max().item()

    t_fp16 = benchmark(run_fp16_linear, args.warmup, args.iters)
    t_lsq = benchmark(run_lsq_quantlinear, args.warmup, args.iters)
    t_int4 = benchmark(run_int4_storage, args.warmup, args.iters)

    print(f"Module: {args.module_name}, input=[{args.batch_size}, {in_features}]")
    print(f"FP16 Linear                : {t_fp16:.3f} ms")
    print(f"LSQ QuantLinear (dequant)  : {t_lsq:.3f} ms")
    print(f"INT4 storage + INT8 compute: {t_int4:.3f} ms")
    print(f"max_abs_err(vs LSQ QuantLinear): {max_abs_err:.6f}")


if __name__ == "__main__":
    main()
