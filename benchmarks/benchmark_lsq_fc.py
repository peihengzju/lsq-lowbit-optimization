import argparse
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

from quant_pipeline.integration import build_lsq_model_from_ckpt, convert_lsq_modules


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


def build_imagenet_val_loader(data_root: str, batch_size: int, num_workers: int):
    val_dir = Path(data_root) / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(str(val_dir), transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def capture_module_input(
    model: torch.nn.Module,
    module_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    loader = build_imagenet_val_loader(data_root, batch_size, num_workers)
    module = get_module_by_name(model, module_name)
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(hook)
    try:
        images, _ = next(iter(loader))
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _ = model(images)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"Failed to capture input for module '{module_name}'")
    return captured[0]


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark LSQ QuantLinear vs INT4-storage INT8-compute")
    parser.add_argument("--lsq-root", type=str, required=True, help="Path to the external LSQ repository root.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--module-name", type=str, default="fc")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--first-last-bits", type=int, default=8)
    parser.add_argument("--disable-first-last-8bit", action="store_true")
    parser.add_argument("--signed-input-first-layer", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=["random", "model-features"], default="random")
    parser.add_argument("--data-root", type=str, default=None)
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
    ).cuda().eval()

    qmodule = get_module_by_name(model, args.module_name)
    if not (hasattr(qmodule, "linear") and hasattr(qmodule, "w_quant") and hasattr(qmodule, "a_quant")):
        raise ValueError(f"Module '{args.module_name}' is not an LSQ QuantLinear")

    in_features = qmodule.linear.in_features
    if args.input_mode == "random":
        x_fp32 = torch.randn(args.batch_size, in_features, device="cuda", dtype=torch.float32)
    else:
        if args.data_root is None:
            raise ValueError("--data-root is required when --input-mode=model-features")
        x_fp32 = capture_module_input(
            model,
            args.module_name,
            args.data_root,
            args.batch_size,
            args.num_workers,
        )
        if x_fp32.dim() != 2:
            raise ValueError(
                f"Captured input for module '{args.module_name}' is rank-{x_fp32.dim()}, expected rank-2"
            )
        x_fp32 = x_fp32.to(device="cuda", dtype=torch.float32)

    x_fp16 = x_fp32.to(torch.float16)

    with torch.no_grad():
        y_ref = qmodule(x_fp32)

    def run_lsq_quantlinear():
        return qmodule(x_fp32)

    fp16_linear = torch.nn.Linear(
        in_features,
        qmodule.linear.out_features,
        bias=qmodule.linear.bias is not None,
    ).cuda().half()
    fp16_linear.weight.data.copy_(qmodule.linear.weight.detach().half())
    if qmodule.linear.bias is not None:
        fp16_linear.bias.data.copy_(qmodule.linear.bias.detach().half())

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
    results = convert_lsq_modules(
        model_int4,
        only_names=[args.module_name],
        convert_linear=True,
        convert_conv=False,
    )
    converted = [item for item in results if item.converted]
    if not converted:
        reasons = ", ".join(f"{item.name}: {item.reason}" for item in results) or "no matching modules"
        raise RuntimeError(f"Module '{args.module_name}' was not converted. {reasons}")
    int4_linear = get_module_by_name(model_int4, args.module_name)

    def run_int4_storage():
        return int4_linear(x_fp32)

    with torch.no_grad():
        y_int4 = run_int4_storage()
        diff = y_ref - y_int4
        max_abs_err = diff.abs().max().item()
        mean_abs_err = diff.abs().mean().item()

    t_fp16 = benchmark(run_fp16_linear, args.warmup, args.iters)
    t_lsq = benchmark(run_lsq_quantlinear, args.warmup, args.iters)
    t_int4 = benchmark(run_int4_storage, args.warmup, args.iters)

    print(f"Module: {args.module_name}, input=[{x_fp32.shape[0]}, {in_features}]")
    print(f"Input mode                  : {args.input_mode}")
    print(f"FP16 Linear                 : {t_fp16:.3f} ms")
    print(f"LSQ QuantLinear (dequant)   : {t_lsq:.3f} ms")
    print(f"INT4 storage + INT8 compute : {t_int4:.3f} ms")
    print(f"max_abs_err(vs LSQ QuantLinear): {max_abs_err:.6f}")
    print(f"mean_abs_err(vs LSQ QuantLinear): {mean_abs_err:.6f}")


if __name__ == "__main__":
    main()
