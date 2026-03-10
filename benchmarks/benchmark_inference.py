import argparse
import time

import torch

from quant_pipeline.ops import int4_int8_gemm, int8_int8_gemm
from quant_pipeline.quantization import pack_int4_weights


def benchmark_once(fn, warmup=20, iters=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def main(args):
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")

    m, n, k = args.m, args.n, args.k
    if k % 2 != 0:
        raise ValueError("k must be even")

    a_int8 = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device)
    b_int8 = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device)
    b_int4 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device)
    b_int4_packed = pack_int4_weights(b_int4)

    # FP16 baseline
    a_fp16 = a_int8.to(torch.float16)
    b_fp16 = b_int8.to(torch.float16)

    def run_fp16():
        return a_fp16 @ b_fp16.t()

    def run_int8():
        return int8_int8_gemm(a_int8, b_int8)

    def run_int4_storage():
        return int4_int8_gemm(a_int8, b_int4_packed)

    fp16_ms = benchmark_once(run_fp16, args.warmup, args.iters)
    int8_ms = benchmark_once(run_int8, args.warmup, args.iters)
    int4_ms = benchmark_once(run_int4_storage, args.warmup, args.iters)

    print(f"Shape: A[{m},{k}] x B[{n},{k}]^T")
    print(f"FP16 torch.matmul      : {fp16_ms:.3f} ms")
    print(f"INT8 custom GEMM       : {int8_ms:.3f} ms")
    print(f"INT4 storage + INT8 GEMM: {int4_ms:.3f} ms")

    # Quick correctness check (small random slice)
    if args.check:
        m_chk = min(64, m)
        n_chk = min(64, n)
        k_chk = min(256, k)

        a_chk = a_int8[:m_chk, :k_chk].cpu().to(torch.int32)
        b_chk = b_int8[:n_chk, :k_chk].cpu().to(torch.int32)
        ref_int8 = a_chk @ b_chk.t()
        out_int8 = int8_int8_gemm(a_int8[:m_chk, :k_chk].contiguous(), b_int8[:n_chk, :k_chk].contiguous()).cpu()
        ok_int8 = torch.equal(ref_int8, out_int8)

        b4_chk = b_int4[:n_chk, :k_chk].cpu().to(torch.int32)
        ref_int4 = a_chk @ b4_chk.t()
        out_int4 = int4_int8_gemm(a_int8[:m_chk, :k_chk].contiguous(), b_int4_packed[:n_chk, : k_chk // 2].contiguous()).cpu()
        ok_int4 = torch.equal(ref_int4, out_int4)

        print(f"INT8 correctness: {ok_int8}")
        print(f"INT4 correctness: {ok_int4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark INT4 storage + INT8 compute pipeline")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args())
