# INT4 CUDA Inference Prototype for Quantized CNNs

Profiling-driven low-bit inference project built with PyTorch C++/CUDA extensions.

## Training repository

This project builds on an LSQ ImageNet reproduction repository:

https://github.com/peihengzju/LSQ-ImageNet-Reproduction

## TL;DR

This repository asks a systems question: why do low-bit quantized models often remain slower than native FP inference? Starting from an LSQ ImageNet reproduction, it builds a custom CUDA inference prototype, converts LSQ checkpoints into custom operators, and uses profiler-driven optimization to reduce the gap between quantized execution and native cuDNN inference.

## Key Contributions

- Built a custom PyTorch CUDA extension for INT4-weight / INT8-activation inference
- Implemented INT4 packed weights and custom GEMM / fused execution paths
- Integrated LSQ training checkpoints into a custom inference backend
- Built a fixed-batch profiling pipeline for identifying runtime bottlenecks
- Reduced converted-path profiling runtime from `15.44s` to `8.15s` through targeted kernel and lowering optimization

## Results

| Configuration | Top-1 | Top-5 | Time |
|---|---:|---:|---:|
| FP native | 69.78 | 89.11 | 56.2s |
| LSQ native | 68.69 | 88.30 | 58.4s |
| Early converted LSQ | 68.75 | 88.38 | 148.5s |
| Best retained converted LSQ | 68.76 | 88.38 | 107.9s |

Fixed-batch converted profiling progression:

| Stage | Wall Time |
|---|---:|
| Initial converted path | 15.44s |
| After fused `3x3` lowering | 9.62s |
| After removing host sync / padding overhead | 8.46s |
| After fused `3x3` conv path | 8.22s |
| Current retained converted profile (`fusedconv_v2`) | 8.15s |

The converted path preserves full-model accuracy and materially improves over the initial converted implementation, but it still does not beat native cuDNN-backed execution end to end. The remaining gap is primarily a systems problem rather than a quantization-accuracy problem.

## Repository Structure

- [`csrc/int4_int8_kernels.cu`](/home/yph3738/projects/cuda_optimization/csrc/int4_int8_kernels.cu): CUDA kernels
- [`csrc/int4_int8_ext.cpp`](/home/yph3738/projects/cuda_optimization/csrc/int4_int8_ext.cpp): PyTorch extension bindings
- [`quant_pipeline/ops/int4_int8_gemm.py`](/home/yph3738/projects/cuda_optimization/quant_pipeline/ops/int4_int8_gemm.py): Python wrappers for custom CUDA ops
- [`quant_pipeline/ops/int4_conv2d.py`](/home/yph3738/projects/cuda_optimization/quant_pipeline/ops/int4_conv2d.py): custom low-bit convolution module
- [`quant_pipeline/ops/int4_linear.py`](/home/yph3738/projects/cuda_optimization/quant_pipeline/ops/int4_linear.py): custom low-bit linear module
- [`quant_pipeline/integration/lsq_adapter.py`](/home/yph3738/projects/cuda_optimization/quant_pipeline/integration/lsq_adapter.py): LSQ checkpoint import and conversion utilities
- [`benchmarks/benchmark_inference.py`](/home/yph3738/projects/cuda_optimization/benchmarks/benchmark_inference.py): synthetic GEMM benchmark
- [`benchmarks/benchmark_lsq_fc.py`](/home/yph3738/projects/cuda_optimization/benchmarks/benchmark_lsq_fc.py): LSQ layer benchmark
- [`scripts/eval_lsq_imagenet.py`](/home/yph3738/projects/cuda_optimization/scripts/eval_lsq_imagenet.py): full-model ImageNet evaluation
- [`scripts/profile_lsq_inference.py`](/home/yph3738/projects/cuda_optimization/scripts/profile_lsq_inference.py): fixed-batch inference profiling
- [`artifacts/evals/`](/home/yph3738/projects/cuda_optimization/artifacts/evals): saved evaluation outputs
- [`artifacts/profiles/`](/home/yph3738/projects/cuda_optimization/artifacts/profiles): saved profiling outputs
- [`docs/reports/`](/home/yph3738/projects/cuda_optimization/docs/reports): report sources and generated PDFs

## Environment

Validated setup:

- Python 3.12
- `torch==2.2.2` with CUDA 12.1
- CUDA Toolkit 12.1
- `gcc-12` / `g++-12`

The CUDA toolkit version must match the PyTorch CUDA build.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Sanity check:

```bash
python -c "import torch; import int4_int8_ext; print('extension ok')"
```

## Reproducing Experiments

### 1. FP Native Evaluation

```bash
python scripts/eval_lsq_imagenet.py \
  --lsq-root /path/to/LSQ \
  --ckpt /path/to/fp_checkpoint.pth \
  --data-root /path/to/imagenet \
  --mode native
```

### 2. LSQ Native vs Converted Evaluation

```bash
python scripts/eval_lsq_imagenet.py \
  --lsq-root /path/to/LSQ \
  --ckpt /path/to/lsq_checkpoint.pth \
  --data-root /path/to/imagenet \
  --mode both \
  --convert-linear \
  --convert-conv
```

### 3. Fixed-Batch Profiling

```bash
python scripts/profile_lsq_inference.py \
  --lsq-root /path/to/LSQ \
  --ckpt /path/to/lsq_checkpoint.pth \
  --data-root /path/to/imagenet \
  --mode converted \
  --convert-linear \
  --convert-conv \
  --batch-size 64 \
  --warmup-batches 5 \
  --active-batches 20 \
  --topk 25 \
  --reuse-single-batch
```

### 4. LSQ Layer Benchmark

```bash
python benchmarks/benchmark_lsq_fc.py \
  --lsq-root /path/to/LSQ \
  --ckpt /path/to/lsq_checkpoint.pth \
  --module-name fc \
  --w-bits 4 \
  --a-bits 4 \
  --disable-first-last-8bit \
  --input-mode model-features \
  --data-root /path/to/imagenet \
  --batch-size 256 \
  --warmup 50 \
  --iters 200
```

## Limitations

- This repository does not distribute pretrained checkpoints.
- LSQ training is maintained in the external LSQ repository used by the experiments.
- The current implementation is a systems prototype, not a production-ready inference framework.
- end-to-end runtime is still slower than native cuDNN-backed inference
- the best-performing retained path converts 19 layers rather than all 21 quantized layers
- residual front-end quantization work still exists
- the current convolution backend is specialized but not yet equivalent to a mature implicit-GEMM or production fused-convolution implementation
- profile-level memory traffic is still higher than native LSQ
