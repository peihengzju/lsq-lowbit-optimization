# LSQ + CUDA Optimization Summary

## Goal

Start from an LSQ ImageNet reproduction and push it toward a systems-oriented low-bit deployment path:

- reproduce LSQ training behavior
- integrate trained LSQ checkpoints into a custom low-bit CUDA path
- analyze why low-bit models are not automatically faster on commodity GPUs

## Current Results

- LSQ baseline on ImageNet val is stable:
  - native LSQ full-model eval: `68.69 / 88.30`
- Converted full-model eval preserves accuracy:
  - converted model: `68.75 / 88.38`
- Converted modules:
  - 19 internal quantized conv layers converted
  - `conv1` and `fc` remain 8-bit and are intentionally skipped

## What Was Implemented

- INT4 packed weight representation
- INT4-storage / INT8-compute CUDA GEMM kernel
- LSQ checkpoint adapter
- full-model evaluation for native vs converted paths
- profiling pipeline for bottleneck analysis

## Profiling Findings

- Layer-level `fc` conversion can speed up LSQ-style quantized inference.
- Full-model conversion does not yet outperform native cuDNN execution.
- Profiling shows the main bottleneck is not the LSQ algorithm itself.
- The main bottlenecks are:
  - convolution lowering overhead
  - many small kernel launches
  - intermediate tensor allocation and movement

## Interpretation

This project already demonstrates a systems-oriented contribution:

- low-bit training results do not automatically translate into low-bit runtime efficiency
- explicit low-bit data layout and CUDA kernels are required
- the limiting factor in the current full-model path is the lowering / integration strategy, not only GEMM arithmetic

## Next Step

Move from `F.unfold + GEMM` toward more fused conv handling:

- special-case `3x3, stride=1, padding=1`
- reduce or eliminate Python-side lowering overhead
- move more of quantize + lowering into CUDA
- eventually replace explicit lowering with implicit GEMM or a fused conv path
