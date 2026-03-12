# INT4 CUDA Inference Prototype for Quantized CNNs

Profiling-driven low-bit inference project built with PyTorch C++/CUDA extensions.

## LSQ Foundation

This project builds on an LSQ ImageNet reproduction repository:

https://github.com/peihengzju/LSQ-ImageNet-Reproduction

That LSQ work is not only a training dependency. It is the correctness baseline for the deployment and backend work in this repository: checkpoint conversion, low-bit operator integration, profiling, and native-vs-converted comparisons all depend on matching the LSQ quantization behavior.

## TL;DR

This repository asks a systems question: why do low-bit quantized models often remain slower than native FP inference? Starting from a verified LSQ ImageNet reproduction, it uses LSQ as the model-quality baseline and conversion target for downstream deployment/backend work, builds a custom CUDA inference prototype, converts LSQ checkpoints into custom operators, and uses profiler-driven optimization to reduce overhead on the converted profiling path.

## Key Contributions

- Built a custom PyTorch CUDA extension for INT4-weight / INT8-activation inference
- Implemented INT4 packed weights and custom GEMM / fused execution paths
- Used LSQ checkpoints as the correctness anchor for downstream deployment / backend experiments
- Built a fixed-batch profiling pipeline for identifying runtime bottlenecks
- Reduced fixed-batch converted profiling runtime from `15.44s` to `8.15s` (`1.89x`) on the retained profiling path through targeted kernel and lowering optimization

## Results

End-to-end evaluation results:

| Configuration | Top-1 | Top-5 | Time |
|---|---:|---:|---:|
| FP native | 69.78 | 89.11 | 56.2s |
| LSQ native | 68.69 | 88.30 | 58.4s |
| Early converted LSQ | 68.75 | 88.38 | 148.5s |
| Best retained converted LSQ | 68.76 | 88.38 | 107.9s |

Fixed-batch converted profiling progression (profiling-only with `--reuse-single-batch`, not end-to-end):

| Stage | Wall Time | Speedup vs Initial |
|---|---:|---:|
| Initial converted path | 15.44s | 1.00x |
| After fused `3x3` lowering | 9.62s | 1.60x |
| After removing host sync / padding overhead | 8.46s | 1.83x |
| After fused `3x3` conv path | 8.22s | 1.88x |
| Current retained converted profile (`fusedconv_v2`) | 8.15s | 1.89x |

The `1.89x` improvement above applies only to the retained fixed-batch profiling path. It should not be interpreted as an end-to-end model speedup: the best retained converted evaluation is still slower than native LSQ (`107.9s` vs `58.4s`) and FP native (`56.2s`). The converted path preserves full-model accuracy and materially improves over the initial converted implementation, but the remaining gap is still primarily a systems problem rather than a quantization-accuracy problem.

## Repository Structure

Core source:

- [`csrc/int4_int8_kernels.cu`](csrc/int4_int8_kernels.cu): CUDA kernels
- [`csrc/int4_int8_ext.cpp`](csrc/int4_int8_ext.cpp): PyTorch extension bindings
- [`quant_pipeline/ops/int4_int8_gemm.py`](quant_pipeline/ops/int4_int8_gemm.py): Python wrappers for custom CUDA ops
- [`quant_pipeline/ops/int4_conv2d.py`](quant_pipeline/ops/int4_conv2d.py): custom low-bit convolution module
- [`quant_pipeline/ops/int4_linear.py`](quant_pipeline/ops/int4_linear.py): custom low-bit linear module
- [`quant_pipeline/quantization/int4_pack.py`](quant_pipeline/quantization/int4_pack.py): INT4 packing utilities
- [`quant_pipeline/integration/lsq_adapter.py`](quant_pipeline/integration/lsq_adapter.py): LSQ checkpoint import and conversion utilities
- [`benchmarks/benchmark_inference.py`](benchmarks/benchmark_inference.py): synthetic GEMM benchmark
- [`benchmarks/benchmark_lsq_fc.py`](benchmarks/benchmark_lsq_fc.py): LSQ layer benchmark
- [`scripts/build_ext.sh`](scripts/build_ext.sh): extension build helper
- [`scripts/eval_lsq_imagenet.py`](scripts/eval_lsq_imagenet.py): full-model ImageNet evaluation
- [`scripts/profile_lsq_inference.py`](scripts/profile_lsq_inference.py): fixed-batch inference profiling

Validation and reports:

- [`tests/test_int4_pack.py`](tests/test_int4_pack.py): INT4 packing tests
- [`docs/reports/summary_lsq_cuda.md`](docs/reports/summary_lsq_cuda.md): short project summary
- [`docs/reports/report_lsq_cuda_cn.md`](docs/reports/report_lsq_cuda_cn.md): detailed Chinese write-up
- [`docs/reports/report_lsq_cuda_cn.pdf`](docs/reports/report_lsq_cuda_cn.pdf): rendered report PDF

Generated experiment outputs:

- [`artifacts/benchmarks/`](artifacts/benchmarks/): saved benchmark notes and outputs
- [`artifacts/evals/`](artifacts/evals/): saved evaluation outputs
- [`artifacts/profiles/`](artifacts/profiles/): saved profiling outputs

Project packaging:

- [`requirements.txt`](requirements.txt): Python dependencies
- [`setup.py`](setup.py): editable-install package definition

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
- The reported `1.89x` speedup is a profiling-path improvement, not an end-to-end model speedup.
- End-to-end runtime is still slower than native cuDNN-backed inference.
- The best-performing retained path converts 19 layers rather than all 21 quantized layers.
- Residual front-end quantization work still exists.
- The current convolution backend is specialized but not yet equivalent to a mature implicit-GEMM or production fused-convolution implementation.
- Profile-level memory traffic is still higher than native LSQ.

## License

MIT License. See [`LICENSE`](LICENSE).
