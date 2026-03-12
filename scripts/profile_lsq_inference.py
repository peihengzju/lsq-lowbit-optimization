import argparse
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.profiler import ProfilerActivity, profile

from quant_pipeline.integration import build_lsq_model_from_ckpt, convert_lsq_modules


def parse_args():
    parser = argparse.ArgumentParser("Profile native vs converted LSQ inference")
    parser.add_argument("--lsq-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--first-last-bits", type=int, default=8)
    parser.add_argument("--disable-first-last-8bit", action="store_true")
    parser.add_argument("--signed-input-first-layer", action="store_true")
    parser.add_argument("--mode", choices=["native", "converted", "both"], default="both")
    parser.add_argument("--convert-linear", action="store_true")
    parser.add_argument("--convert-conv", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--active-batches", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--reuse-single-batch", action="store_true")
    return parser.parse_args()


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


def load_single_batch(loader, device: torch.device) -> torch.Tensor:
    images, _ = next(iter(loader))
    return images.to(device, non_blocking=True)


def build_model(args):
    model = build_lsq_model_from_ckpt(
        args.lsq_root,
        args.ckpt,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        disable_first_last_8bit=args.disable_first_last_8bit,
        signed_input_first_layer=args.signed_input_first_layer,
    )
    return model


def maybe_convert(model, args):
    if args.mode == "native":
        return model, []
    results = convert_lsq_modules(
        model,
        convert_linear=args.convert_linear,
        convert_conv=args.convert_conv,
    )
    return model, results


def print_conversion_summary(results):
    converted = [item.name for item in results if item.converted]
    skipped = [f"{item.name} ({item.reason})" for item in results if not item.converted]
    print(f"Converted modules: {len(converted)}")
    if converted:
        print("  " + ", ".join(converted))
    print(f"Skipped modules: {len(skipped)}")
    if skipped:
        print("  " + "; ".join(skipped))


def run_profile(tag: str, model: torch.nn.Module, loader, device: torch.device, args):
    model.eval().to(device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    total_batches = args.warmup_batches + args.active_batches
    trace_dir = None
    if args.trace_dir is not None:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

    fixed_images = load_single_batch(loader, device) if args.reuse_single_batch else None

    start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loader_iter = iter(loader)
        for idx in range(total_batches):
            if fixed_images is None:
                images, _ = next(loader_iter)
                images = images.to(device, non_blocking=True)
            else:
                images = fixed_images
            with torch.no_grad():
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if idx >= args.warmup_batches:
                prof.step()

    elapsed = time.perf_counter() - start

    if trace_dir is not None:
        trace_path = trace_dir / f"{tag}_trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace saved to {trace_path}")

    key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(
        sort_by=key,
        row_limit=args.topk,
    )
    print(f"\n== {tag} ==")
    print(
        f"Profiled batches: warmup={args.warmup_batches}, active={args.active_batches}, "
        f"wall_time={elapsed:.2f}s"
    )
    print(table)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_imagenet_val_loader(args.data_root, args.batch_size, args.num_workers)

    if args.mode in ("native", "both"):
        native_model = build_model(args)
        run_profile("native", native_model, loader, device, args)

    if args.mode in ("converted", "both"):
        converted_model = build_model(args)
        converted_model, results = maybe_convert(converted_model, args)
        print_conversion_summary(results)
        run_profile("converted", converted_model, loader, device, args)


if __name__ == "__main__":
    main()
