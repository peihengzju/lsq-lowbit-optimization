# lsq-lowbit-optimization

LSQ-centered low-bit optimization project for PyTorch, with a continuous roadmap from hardware-aware inference optimization to training-time optimization.

## Scope

Current focus (implemented):
- INT4 weight storage (packed `uint8`, 2 values per byte)
- INT4 on-the-fly unpack in CUDA
- INT8 x INT8 -> INT32 GEMM path with `dp4a`
- Layer-level comparison against native LSQ `QuantLinear`

Planned next phase:
- Training-side low-bit optimization built on the same LSQ pipeline
- Better train/infer consistency for scale handling and quantization behavior

## Why this project

- Keep one continuous research/engineering thread: LSQ training artifacts -> low-bit deployment path.
- Bridge algorithm and systems work: quantization-aware representation + CUDA kernel implementation.
- Provide reproducible scripts for benchmark and integration validation.

## Resume-Friendly Highlights

- Built a custom PyTorch CUDA extension for INT4-storage/INT8-compute inference.
- Implemented packed INT4 handling and fused low-bit GEMM kernel path.
- Integrated external LSQ checkpoints into a hardware-aware inference pipeline.
- Added benchmark flows for synthetic GEMM and LSQ module-level latency/error comparison.

## Repository Structure

- `csrc/int4_int8_kernels.cu`: CUDA kernels (`int8_int8_gemm`, `int4_int8_gemm`)
- `csrc/int4_int8_ext.cpp`: PyTorch C++ extension bindings
- `quant_pipeline/quantization/int4_pack.py`: INT4 pack/unpack utilities
- `quant_pipeline/ops/int4_int8_gemm.py`: Python wrapper for custom op
- `quant_pipeline/ops/int4_linear.py`: `Int4WeightInt8ActLinear`
- `quant_pipeline/integration/lsq_adapter.py`: LSQ model conversion utilities
- `benchmarks/benchmark_inference.py`: synthetic GEMM benchmark
- `benchmarks/benchmark_lsq_fc.py`: LSQ layer benchmark

## Environment

Validated setup:
- `torch==2.2.2` (CUDA 12.1 build)
- CUDA Toolkit `12.1`
- `gcc/g++ 12`

CUDA toolkit and PyTorch CUDA runtime versions must match.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Build extension (explicit toolchain recommended):

```bash
CUDA_HOME=/usr/local/cuda-12.1 \
CUDACXX=/usr/local/cuda-12.1/bin/nvcc \
CC=/usr/bin/gcc-12 \
CXX=/usr/bin/g++-12 \
CUDAHOSTCXX=/usr/bin/g++-12 \
FORCE_CUDA=1 \
pip install -e . --no-build-isolation --force-reinstall
```

Sanity check:

```bash
python -c "import int4_int8_ext; print('extension ok')"
```

## Benchmarks

Synthetic GEMM:

```bash
python benchmarks/benchmark_inference.py \
  --m 1024 --n 1024 --k 1024 --iters 200 --check
```

LSQ layer benchmark (bring your own LSQ repo/checkpoint):

```bash
python benchmarks/benchmark_lsq_fc.py \
  --lsq-root /path/to/LSQ \
  --ckpt /path/to/checkpoint.pth \
  --module-name fc \
  --w-bits 4 --a-bits 4 --disable-first-last-8bit \
  --batch-size 256 --warmup 50 --iters 200
```

Outputs include latency comparison and numerical error metrics (for example `max_abs_err` vs LSQ baseline layer output).

## Model and Checkpoint Policy

- Model weights/checkpoints are intentionally not included in this repository.
- Users should download or prepare LSQ checkpoints themselves, based on their own license/access constraints.
- This repository provides the integration and benchmarking pipeline, not pretrained model distribution.

## LSQ Integration Notes

- LSQ training is not implemented in this repository yet.
- Current integration reuses trained LSQ checkpoint weights/scales and converts selected `QuantLinear` layers.
- Default conversion targets signed INT4 weight layers (`qn=-8`, `qp=7`).

## Common Issues

1. `The detected CUDA version mismatches the version that was used to compile PyTorch`
- Use CUDA 12.1 toolkit with torch 2.2.2 cu121.

2. `unsupported GNU version` from nvcc
- Use `gcc-12/g++-12` and pass `CC/CXX/CUDAHOSTCXX` explicitly.

3. `ImportError: libc10.so: cannot open shared object file`
- Import `torch` before extension import, or ensure torch shared libraries are visible in your environment.

4. `Module 'fc' was not converted`
- The target layer may not be 4-bit (for example first/last layer kept at 8-bit).
- Try `--disable-first-last-8bit` or choose a 4-bit `QuantLinear` layer.

## Roadmap

- [x] Hardware-aware INT4/INT8 inference path and validation
- [x] LSQ checkpoint adapter for layer conversion and benchmarking
- [ ] Training-time optimization integrated into the same LSQ low-bit pipeline
