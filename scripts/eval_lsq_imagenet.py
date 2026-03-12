import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from quant_pipeline.integration import build_lsq_model_from_ckpt, convert_lsq_modules


def parse_args():
    parser = argparse.ArgumentParser("Evaluate native LSQ and converted model on ImageNet val")
    parser.add_argument("--lsq-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--first-last-bits", type=int, default=8)
    parser.add_argument("--disable-first-last-8bit", action="store_true")
    parser.add_argument("--signed-input-first-layer", action="store_true")
    parser.add_argument("--mode", choices=["native", "converted", "both"], default="both")
    parser.add_argument("--convert-linear", action="store_true")
    parser.add_argument("--convert-conv", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k)
    return results


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: int | None = None):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    batch_count = 0
    t0 = time.perf_counter()

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        top1, top5 = accuracy(output, target)

        batch_size = images.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_top1 += top1.item()
        total_top5 += top5.item()
        batch_count += 1

        if max_batches is not None and batch_count >= max_batches:
            break

    elapsed = time.perf_counter() - t0
    return {
        "loss": total_loss / max(total_samples, 1),
        "top1": 100.0 * total_top1 / max(total_samples, 1),
        "top5": 100.0 * total_top5 / max(total_samples, 1),
        "samples": total_samples,
        "batches": batch_count,
        "seconds": elapsed,
    }


def build_model(args):
    return build_lsq_model_from_ckpt(
        args.lsq_root,
        args.ckpt,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        disable_first_last_8bit=args.disable_first_last_8bit,
        signed_input_first_layer=args.signed_input_first_layer,
    )


def summarize_conversion(results):
    converted = [item.name for item in results if item.converted]
    skipped = [f"{item.name} ({item.reason})" for item in results if not item.converted]
    print(f"Converted modules: {len(converted)}")
    if converted:
        print("  " + ", ".join(converted))
    print(f"Skipped modules: {len(skipped)}")
    if skipped:
        print("  " + "; ".join(skipped))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_imagenet_val_loader(args.data_root, args.batch_size, args.num_workers)

    run_native = args.mode in ("native", "both")
    run_converted = args.mode in ("converted", "both")

    if run_native:
        native_model = build_model(args).to(device)
        stats = evaluate(native_model, loader, device, args.max_batches)
        print(
            "native "
            f"val_loss={stats['loss']:.4f} val_top1={stats['top1']:.2f} "
            f"val_top5={stats['top5']:.2f} samples={stats['samples']} "
            f"time={stats['seconds']:.1f}s"
        )

    if run_converted:
        converted_model = build_model(args)
        results = convert_lsq_modules(
            converted_model,
            convert_linear=args.convert_linear,
            convert_conv=args.convert_conv,
        )
        summarize_conversion(results)
        converted_model = converted_model.to(device)
        stats = evaluate(converted_model, loader, device, args.max_batches)
        print(
            "converted "
            f"val_loss={stats['loss']:.4f} val_top1={stats['top1']:.2f} "
            f"val_top5={stats['top5']:.2f} samples={stats['samples']} "
            f"time={stats['seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()
