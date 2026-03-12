# Custom INT4 CUDA Inference Backend for Quantized CNNs

Profiling-driven low-bit inference project built with PyTorch C++/CUDA extensions.

This repository focuses on a practical systems question: after training a quantized CNN, what is required to turn that model into a hardware-aware inference path rather than just a quantized model evaluated through standard PyTorch execution?

## Project Summary

The current implementation is built around LSQ-trained ImageNet checkpoints and extends them into a custom low-bit CUDA inference path:

- INT4 packed weight storage
- INT8 compute path based on custom PyTorch CUDA extensions
- checkpoint adapter for converting LSQ quantized layers into custom operators
- full-model ImageNet evaluation for native LSQ vs converted execution
- fixed-batch profiling workflow for identifying real runtime bottlenecks

The emphasis is not only on numerical correctness, but on analyzing why low-bit models are often still slower than native execution and then reducing that gap with targeted kernel and lowering optimizations.

## Why This Project Matters

Low-bit quantization does not automatically imply low-bit efficiency.

In practice, quantized models are often bottlenecked by:

- explicit lowering (`im2col` / `unfold`)
- temporary tensor allocation
- dtype conversion
- repeated quantize/dequantize work
- fragmented CUDA kernel launches

This repository is an attempt to bridge that gap by moving the bottlenecks into custom CUDA code rather than stopping at algorithm-level quantization.

## Key Technical Work

- Built a custom PyTorch CUDA extension for low-bit inference
- Implemented INT4 packed weights with on-the-fly unpacking
- Implemented INT4-weight / INT8-activation GEMM and fused execution paths
- Added specialized fused paths for dominant convolution cases
- Integrated external LSQ checkpoints into a custom inference backend
- Added full-model ImageNet evaluation and fixed-batch profiling scripts
- Used profiler output to drive successive kernel/lowering revisions

## Current Best Retained Results

The table below reflects the retained optimization path. A later `full-cover` experiment that converted `conv1` and `fc` was evaluated and then rolled back because it regressed runtime; it is intentionally excluded from the main results here.

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

Takeaway:

- the converted path preserves full-model accuracy
- the retained CUDA work materially reduces runtime relative to the initial converted implementation
- the project still does not beat native cuDNN-backed execution end to end
- profiling shows that the remaining gap is primarily a systems problem, not a quantization-accuracy problem

## What Is Implemented

Implemented and retained:

- LSQ checkpoint loading and reconstruction
- conversion of eligible middle quantized layers into custom operators
- packed INT4 weight path
- custom INT4-weight / INT8-activation GEMM
- specialized `3x3, stride=1, padding=1` path
- specialized `1x1` path
- fused low-bit linear path
- ImageNet evaluation scripts
- profiler-driven analysis workflow

Not implemented as a final production solution:

- training inside this repository
- end-to-end integer activation pipeline across the whole model
- implicit-GEMM-style final convolution backend
- best-possible kernel tuning comparable to cuDNN or production inference libraries

## Scope

This repository should be viewed as a systems prototype for quantized CNN inference rather than a general-purpose deployment framework.

What is general in the current codebase:

- INT4 packed weight storage
- custom CUDA kernels and PyTorch extension bindings
- fused low-bit linear and selected convolution paths
- profiler-driven workflow for identifying end-to-end bottlenecks

What is still LSQ-specific in the current codebase:

- checkpoint import and model reconstruction flow
- evaluation scripts and naming
- current end-to-end validation on LSQ-trained PreAct-ResNet18/ImageNet

## Repository Layout

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

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Build the extension:

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
python -c "import torch; import int4_int8_ext; print('extension ok')"
```

## Reproducing the Main Experiments

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

## Important Notes

- This repository does not distribute pretrained checkpoints.
- LSQ training is maintained in the external LSQ repository used by the experiments.
- The best retained performance path in this repository intentionally keeps `conv1` and `fc` on the native path because the attempted full-coverage conversion regressed runtime.
- The current implementation should be viewed as a systems prototype, not as a production-ready low-bit inference framework.

## Known Limitations

- end-to-end runtime is still slower than native cuDNN-backed inference
- the best-performing retained path converts 19 layers rather than all 21 quantized layers
- residual front-end quantization work still exists
- the current convolution backend is specialized but not yet equivalent to a mature implicit-GEMM or production fused-convolution implementation
- profile-level memory traffic is still higher than native LSQ

## What This Repository Demonstrates

The repository is intended to document a complete technical thread rather than a single result:

- integrating external quantized checkpoints into a custom inference backend
- implementing PyTorch C++/CUDA extensions for low-bit execution
- using profiling to identify the dominant runtime bottlenecks
- iterating on lowering and kernel design based on measured behavior
- validating that low-bit accuracy can be retained after operator conversion
- retaining unsuccessful directions when they help clarify the remaining gap to native execution

In that sense, the value of the project is not limited to the final numbers. It captures the engineering process required to move from a quantized checkpoint to a more hardware-aware execution path.

## References in This Repo

- Main report: [`docs/reports/report_lsq_cuda_cn.pdf`](/home/yph3738/projects/cuda_optimization/docs/reports/report_lsq_cuda_cn.pdf)
- Supporting report source: [`docs/reports/report_lsq_cuda_cn.tex`](/home/yph3738/projects/cuda_optimization/docs/reports/report_lsq_cuda_cn.tex)
- Example retained converted evaluation: [`artifacts/evals/results_lsq4_fusedconv_v1.txt`](/home/yph3738/projects/cuda_optimization/artifacts/evals/results_lsq4_fusedconv_v1.txt)
- Example retained converted profile: [`artifacts/profiles/profile_lsq4_fusedconv_v2.txt`](/home/yph3738/projects/cuda_optimization/artifacts/profiles/profile_lsq4_fusedconv_v2.txt)
