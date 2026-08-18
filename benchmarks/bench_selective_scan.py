#!/usr/bin/env python3
"""Microbenchmark for selective_scan_cuda fwd/bwd kernels.

Measures two levels:
  1. "e2e"  — selective_scan_fn (Python autograd → C++ → kernel), fwd+bwd as a unit
  2. "cuda" — selective_scan_cuda.fwd/bwd directly (C++ dispatch + kernel only)

The stable ABI transform only changes the C++ dispatch in selective_scan.cpp.
The CUDA kernels use raw pointers via SSMParamsBase — PTX is unchanged.
Expected result: 0% difference. Any >2% on large configs means we broke something
(accidental copy, extra allocation, type conversion).

Usage:
    CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_selective_scan.py
    CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_selective_scan.py --json baseline.json
    CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_selective_scan.py --compare baseline.json
    CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_selective_scan.py --selfcheck
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from triton.testing import do_bench

try:
    import selective_scan_cuda  # noqa: F401 — triggers op registration
    _has_selective_scan_cuda = True
except ImportError:
    _has_selective_scan_cuda = False

_fwd = torch.ops.selective_scan.fwd if _has_selective_scan_cuda and hasattr(torch.ops, "selective_scan") else None
_bwd = torch.ops.selective_scan.bwd if _fwd else None

class _SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        out, x, *rest = _fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        if ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            return rest[0], x
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out, x

    @staticmethod
    def backward(ctx, dout, _dx):
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.has_z:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        else:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z, out = None, None
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = _bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None,
            ctx.delta_softplus, False)
        return du, ddelta, dA, dB, dC, dD, rest[0] if ctx.has_z else None, ddelta_bias, None

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    out, x = _SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    last_state = x[:, :, -1, 1::2]
    return (out, last_state) if return_last_state else out

_has_stable_ops = (
    _has_selective_scan_cuda
    and hasattr(torch.ops, "selective_scan")
    and hasattr(torch.ops.selective_scan, "fwd")
)

WARMUP_MS = 1000
REP_MS = 3000
N_TRIALS = 3

CONFIGS = [
    # (label, batch, dim, seqlen, dstate, n_groups, dtype)
    ("small-fp32",   2,   64,  1024,  16, 1, torch.float32),
    ("small-fp16",   2,   64,  1024,  16, 1, torch.float16),
    ("small-bf16",   2,   64,  1024,  16, 1, torch.bfloat16),
    ("med-fp32",     4,  256,  2048,  16, 1, torch.float32),
    ("med-fp16",     4,  256,  2048,  16, 1, torch.float16),
    ("med-bf16",     4,  256,  2048,  16, 1, torch.bfloat16),
    ("large-fp16",   8,  768,  4096,  16, 1, torch.float16),
    ("large-bf16",   8,  768,  4096,  16, 1, torch.bfloat16),
    ("xlarge-bf16", 16, 1024,  4096,  16, 1, torch.bfloat16),
    ("groups-bf16",  4,  256,  2048,  16, 4, torch.bfloat16),
]


def make_inputs(batch, dim, seqlen, dstate, n_groups, itype, device="cuda"):
    return dict(
        u=torch.randn(batch, dim, seqlen, device=device, dtype=itype),
        delta=0.5 * torch.rand(batch, dim, seqlen, device=device, dtype=itype),
        A=-0.5 * torch.rand(dim, dstate, device=device, dtype=torch.float32),
        B=torch.randn(batch, n_groups, dstate, seqlen, device=device, dtype=itype),
        C=torch.randn(batch, n_groups, dstate, seqlen, device=device, dtype=itype),
        D=torch.randn(dim, device=device, dtype=torch.float32),
        z=torch.randn(batch, dim, seqlen, device=device, dtype=itype),
        delta_bias=0.5 * torch.rand(dim, device=device, dtype=torch.float32),
    )


def bench(fn, grad_to_none=None):
    """Run N_TRIALS, return median of medians and spread."""
    medians = []
    for _ in range(N_TRIALS):
        med = do_bench(fn, warmup=WARMUP_MS, rep=REP_MS,
                       grad_to_none=grad_to_none, return_mode="median")
        medians.append(med)
    medians.sort()
    median_of_medians = medians[N_TRIALS // 2]
    spread_pct = (medians[-1] - medians[0]) / median_of_medians * 100 if median_of_medians > 0 else 0
    return median_of_medians, spread_pct


def gpu_warmup():
    a = torch.randn(2048, 2048, device="cuda")
    for _ in range(200):
        torch.mm(a, a)
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def run_benchmarks():
    results = []
    gpu_name = torch.cuda.get_device_name(0)
    gpu_warmup()

    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Trials: {N_TRIALS}, each {WARMUP_MS}ms warmup + {REP_MS}ms measure, median of medians")
    print()

    # --- Level 1: end-to-end fwd+bwd (what users actually run) ---
    print("=== e2e: selective_scan_fn fwd+bwd ===")
    hdr = f"{'config':<16} {'fwd+bwd':>8} {'var%':>5}"
    print(hdr)
    print("-" * len(hdr))

    for label, batch, dim, seqlen, dstate, n_groups, itype in CONFIGS:
        inputs = make_inputs(batch, dim, seqlen, dstate, n_groups, itype)
        for v in inputs.values():
            v.requires_grad_(True)
        grads = list(inputs.values())

        def fwd_bwd():
            out, _state = selective_scan_fn(
                inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
                inputs["D"], z=inputs["z"], delta_bias=inputs["delta_bias"],
                delta_softplus=True, return_last_state=True,
            )
            out.sum().backward()

        ms, var = bench(fwd_bwd, grad_to_none=grads)
        print(f"{label:<16} {ms:>8.3f} {var:>4.1f}%")

        results.append({
            "config": label, "batch": batch, "dim": dim, "seqlen": seqlen,
            "dstate": dstate, "n_groups": n_groups, "dtype": str(itype),
            "e2e_ms": round(ms, 4),
            "gpu": gpu_name, "pytorch": torch.__version__,
        })

        for v in inputs.values():
            if v.grad is not None:
                v.grad = None
            v.requires_grad_(False)

    # --- Level 2: C++ dispatch layer only (what the transform changes) ---
    if _has_stable_ops:
        print()
        print("=== cuda: selective_scan_cuda.fwd/bwd direct ===")
        hdr2 = f"{'config':<16} {'fwd':>8} {'var%':>5}  {'bwd':>8} {'var%':>5}"
        print(hdr2)
        print("-" * len(hdr2))

        for i, (label, batch, dim, seqlen, dstate, n_groups, itype) in enumerate(CONFIGS):
            inputs = make_inputs(batch, dim, seqlen, dstate, n_groups, itype)
            I = inputs

            def cuda_fwd():
                return torch.ops.selective_scan.fwd(
                    I["u"], I["delta"], I["A"], I["B"], I["C"],
                    I["D"], I["z"], I["delta_bias"], True,
                )

            fwd_ms, fwd_var = bench(cuda_fwd)

            out, x, out_z = torch.ops.selective_scan.fwd(
                I["u"], I["delta"], I["A"], I["B"], I["C"],
                I["D"], I["z"], I["delta_bias"], True,
            )
            dout = torch.randn_like(out)

            def cuda_bwd():
                return torch.ops.selective_scan.bwd(
                    I["u"], I["delta"], I["A"], I["B"], I["C"],
                    I["D"], I["z"], I["delta_bias"], dout, x, out,
                    None, True, False,
                )

            bwd_ms, bwd_var = bench(cuda_bwd)

            print(f"{label:<16} {fwd_ms:>8.3f} {fwd_var:>4.1f}%  {bwd_ms:>8.3f} {bwd_var:>4.1f}%")

            results[i]["cuda_fwd_ms"] = round(fwd_ms, 4)
            results[i]["cuda_bwd_ms"] = round(bwd_ms, 4)

            del out, x, out_z, dout

    return results


def compare(baseline_path, current):
    with open(baseline_path) as f:
        baseline = json.load(f)
    base_map = {r["config"]: r for r in baseline}

    print(f"\n=== e2e comparison ===")
    print(f"{'config':<16} {'base':>8} {'now':>8} {'diff%':>7}")
    print("-" * 42)
    for r in current:
        b = base_map.get(r["config"])
        if not b or "e2e_ms" not in b:
            continue
        p = (r["e2e_ms"] - b["e2e_ms"]) / b["e2e_ms"] * 100 if b["e2e_ms"] else 0
        flag = "  " if abs(p) < 2 else (" *" if abs(p) < 5 else "!!")
        print(f"{r['config']:<16} {b['e2e_ms']:>8.3f} {r['e2e_ms']:>8.3f} {p:>+6.1f}%{flag}")

    if any("cuda_fwd_ms" in r for r in current):
        print(f"\n=== cuda dispatch comparison ===")
        print(f"{'config':<16} {'fwd_base':>8} {'fwd_now':>8} {'fwd%':>7}  "
              f"{'bwd_base':>8} {'bwd_now':>8} {'bwd%':>7}")
        print("-" * 78)
        for r in current:
            b = base_map.get(r["config"])
            if not b or "cuda_fwd_ms" not in b or "cuda_fwd_ms" not in r:
                continue
            fp = (r["cuda_fwd_ms"] - b["cuda_fwd_ms"]) / b["cuda_fwd_ms"] * 100 if b["cuda_fwd_ms"] else 0
            bp = (r["cuda_bwd_ms"] - b["cuda_bwd_ms"]) / b["cuda_bwd_ms"] * 100 if b["cuda_bwd_ms"] else 0
            ff = "  " if abs(fp) < 2 else (" *" if abs(fp) < 5 else "!!")
            bf = "  " if abs(bp) < 2 else (" *" if abs(bp) < 5 else "!!")
            print(f"{r['config']:<16} {b['cuda_fwd_ms']:>8.3f} {r['cuda_fwd_ms']:>8.3f} {fp:>+6.1f}%{ff}  "
                  f"{b['cuda_bwd_ms']:>8.3f} {r['cuda_bwd_ms']:>8.3f} {bp:>+6.1f}%{bf}")


def selfcheck():
    """Run twice and compare to establish the noise floor."""
    print("=== Self-check: measuring noise floor ===")
    print("Running benchmark twice with identical code...\n")

    run1 = run_benchmarks()
    print("\n--- Second run ---\n")
    run2 = run_benchmarks()

    print("\n=== Noise floor (run1 vs run2) ===")
    print(f"{'config':<16} {'run1':>8} {'run2':>8} {'diff%':>7}  note")
    print("-" * 55)
    for r1, r2 in zip(run1, run2):
        p = (r2["e2e_ms"] - r1["e2e_ms"]) / r1["e2e_ms"] * 100 if r1["e2e_ms"] else 0
        note = "noise" if abs(p) < 2 else "UNSTABLE" if abs(p) > 5 else "marginal"
        print(f"{r1['config']:<16} {r1['e2e_ms']:>8.3f} {r2['e2e_ms']:>8.3f} {p:>+6.1f}%  {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--compare", type=str, default=None, help="Compare against baseline JSON")
    parser.add_argument("--selfcheck", action="store_true", help="Run twice to measure noise floor")
    args = parser.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    results = run_benchmarks()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.json}")

    if args.compare:
        compare(args.compare, results)


if __name__ == "__main__":
    main()
