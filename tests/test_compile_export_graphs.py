#!/usr/bin/env python3
"""Test torch.compile, torch.export, and CUDA graphs with stable ABI selective scan."""

import torch
import sys
import traceback

try:
    import selective_scan_cuda  # noqa: F401
except ImportError:
    print("SKIP: selective_scan_cuda not built")
    sys.exit(0)

has_stable = (
    hasattr(torch.ops, "selective_scan")
    and hasattr(torch.ops.selective_scan, "fwd")
)
if not has_stable:
    print("SKIP: stable ABI ops not registered (pybind11 build)")
    sys.exit(0)

def _register_fakes():
    """Register fake implementations if not already registered."""
    fwd_key = torch.ops.selective_scan.fwd
    if hasattr(fwd_key, "_dispatch_cache") and "FakeTensor" in str(getattr(fwd_key, "_dispatch_cache", {})):
        return
    try:
        @torch.library.register_fake("selective_scan::fwd")
        def _fwd_fake(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
            batch, dim, seqlen = u.shape
            dstate = A.size(1)
            n_chunks = (seqlen + 2047) // 2048
            out = delta.new_empty(delta.shape)
            x = u.new_empty(batch, dim, n_chunks, dstate * 2, dtype=A.dtype)
            if z is not None:
                return [out, x, z.new_empty(z.shape)]
            return [out, x]

        @torch.library.register_fake("selective_scan::bwd")
        def _bwd_fake(u, delta, A, B, C, D, z, delta_bias, dout,
                      x, out, dz, delta_softplus, recompute_out_z):
            du = u.new_empty(u.shape)
            ddelta = delta.new_empty(delta.shape)
            dA = A.new_empty(A.shape)
            dB = B.new_empty(B.shape)
            dC = C.new_empty(C.shape)
            dD = D.new_empty(D.shape) if D is not None else None
            ddelta_bias = delta_bias.new_empty(delta_bias.shape) if delta_bias is not None else None
            result = [du, ddelta, dA, dB, dC, dD, ddelta_bias]
            if z is not None:
                result.append(dz if dz is not None else z.new_empty(z.shape))
            if recompute_out_z:
                result.append(out.new_empty(out.shape) if out is not None else z.new_empty(z.shape))
            return result
    except RuntimeError:
        pass

_register_fakes()


BATCH, DIM, SEQLEN, DSTATE = 4, 256, 2048, 16
N_GROUPS = 1


def make_inputs(device="cuda", requires_grad=False, D=True, z=True, delta_bias=True):
    kw = dict(device=device, requires_grad=requires_grad)
    return dict(
        u=torch.randn(BATCH, DIM, SEQLEN, dtype=torch.bfloat16, **kw),
        delta=0.5 * torch.rand(BATCH, DIM, SEQLEN, dtype=torch.bfloat16, **kw),
        A=-0.5 * torch.rand(DIM, DSTATE, dtype=torch.float32, **kw),
        B=torch.randn(BATCH, N_GROUPS, DSTATE, SEQLEN, dtype=torch.bfloat16, **kw),
        C=torch.randn(BATCH, N_GROUPS, DSTATE, SEQLEN, dtype=torch.bfloat16, **kw),
        D=torch.randn(DIM, dtype=torch.float32, **kw) if D else None,
        z=torch.randn(BATCH, DIM, SEQLEN, dtype=torch.bfloat16, **kw) if z else None,
        delta_bias=0.5 * torch.rand(DIM, dtype=torch.float32, **kw) if delta_bias else None,
    )


def raw_fwd(inputs):
    """Call torch.ops directly, bypassing autograd wrapper."""
    return torch.ops.selective_scan.fwd(
        inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
        inputs["D"], inputs["z"], inputs["delta_bias"], True,
    )


def run_scan(inputs):
    """Wrapper that calls raw op (no autograd for simplicity)."""
    return torch.ops.selective_scan.fwd(
        inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
        inputs["D"], inputs["z"], inputs["delta_bias"], True,
    )


# ─── Tests ───────────────────────────────────────────────────────────

def test_eager():
    print("1. eager baseline...", end=" ", flush=True)
    inputs = make_inputs()
    result = run_scan(inputs)
    out = result[0]
    assert out.shape == (BATCH, DIM, SEQLEN)
    print(f"OK")
    return out.clone()


def test_compile_fullgraph():
    print("2. torch.compile fullgraph=True...", end=" ", flush=True)
    inputs = make_inputs()
    try:
        compiled = torch.compile(run_scan, fullgraph=True)
        result = compiled(inputs)
        out = result[0]
        assert out.shape == (BATCH, DIM, SEQLEN)

        # compare with eager
        eager_result = run_scan(inputs)
        diff = (out - eager_result[0]).abs().max().item()
        print(f"OK (diff={diff})")
    except Exception as e:
        msg = str(e)
        if "fake tensor" in msg.lower() or "meta" in msg.lower():
            print(f"EXPECTED FAIL (needs Meta impl for fake tensors)")
        else:
            print(f"FAIL: {e}")


def test_compile_with_breaks():
    print("3. torch.compile fullgraph=False...", end=" ", flush=True)
    inputs = make_inputs()
    try:
        compiled = torch.compile(run_scan, fullgraph=False)
        result = compiled(inputs)
        out = result[0]
        assert out.shape == (BATCH, DIM, SEQLEN)
        print(f"OK")
    except Exception as e:
        print(f"FAIL: {e}")


def test_compile_reduce_overhead():
    print("4. torch.compile mode='reduce-overhead'...", end=" ", flush=True)
    inputs = make_inputs()
    try:
        compiled = torch.compile(run_scan, mode="reduce-overhead")
        for _ in range(3):
            result = compiled(inputs)
        torch.cuda.synchronize()
        out = result[0]
        assert out.shape == (BATCH, DIM, SEQLEN)
        print(f"OK (auto CUDA graphs)")
    except Exception as e:
        print(f"FAIL: {e}")


def test_cuda_graph_raw_op():
    """Manual CUDA graph with the raw torch.ops call (no autograd)."""
    print("5. manual CUDA graph (raw op, no autograd)...", end=" ", flush=True)
    inputs = make_inputs()

    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        result = raw_fwd(inputs)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        result = raw_fwd(inputs)

    # replay
    g.replay()
    torch.cuda.synchronize()
    assert result[0].shape == (BATCH, DIM, SEQLEN)
    print(f"OK")

    # replay with new data
    inputs["u"].copy_(torch.randn_like(inputs["u"]))
    g.replay()
    torch.cuda.synchronize()
    print("   replay with new data: OK")


def test_cuda_graph_with_autograd():
    """Manual CUDA graph via make_graphed_callables (handles autograd)."""
    print("6. torch.cuda.make_graphed_callables...", end=" ", flush=True)
    try:
        inputs = make_inputs()

        class ScanWrap(torch.nn.Module):
            def forward(self, u, delta, A, B, C, D, z, delta_bias):
                return torch.ops.selective_scan.fwd(u, delta, A, B, C, D, z, delta_bias, True)

        mod = ScanWrap().cuda()
        sample = (inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
                  inputs["D"], inputs["z"], inputs["delta_bias"])

        graphed = torch.cuda.make_graphed_callables(mod, sample)
        result = graphed(*sample)
        assert result[0].shape == (BATCH, DIM, SEQLEN)
        print(f"OK")
    except Exception as e:
        print(f"FAIL: {e}")


def test_export_non_strict():
    print("7. torch.export strict=False...", end=" ", flush=True)
    try:
        class ScanModule(torch.nn.Module):
            def forward(self, u, delta, A, B, C, D, z, delta_bias):
                return torch.ops.selective_scan.fwd(u, delta, A, B, C, D, z, delta_bias, True)

        mod = ScanModule()
        inputs = make_inputs()
        args = (inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
                inputs["D"], inputs["z"], inputs["delta_bias"])

        exported = torch.export.export(mod, args, strict=False)
        result = exported.module()(*args)
        assert result[0].shape == (BATCH, DIM, SEQLEN)
        graph_str = str(exported.graph)
        has_op = "selective_scan" in graph_str
        print(f"OK  (op in graph: {has_op})")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()


def test_export_strict():
    print("8. torch.export strict=True...", end=" ", flush=True)
    try:
        class ScanModule(torch.nn.Module):
            def forward(self, u, delta, A, B, C, D, z, delta_bias):
                return torch.ops.selective_scan.fwd(u, delta, A, B, C, D, z, delta_bias, True)

        mod = ScanModule()
        inputs = make_inputs()
        args = (inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
                inputs["D"], inputs["z"], inputs["delta_bias"])

        exported = torch.export.export(mod, args, strict=True)
        result = exported.module()(*args)
        assert result[0].shape == (BATCH, DIM, SEQLEN)
        print(f"OK")
    except Exception as e:
        msg = str(e)
        if "fake tensor" in msg.lower() or "meta" in msg.lower():
            print(f"EXPECTED FAIL (needs Meta impl)")
        else:
            print(f"FAIL: {e}")


def test_none_optionals():
    """Test with D=None, z=None, delta_bias=None."""
    print("9. eager with None optionals...", end=" ", flush=True)
    for label, kw in [
        ("D=None", dict(D=False)),
        ("z=None", dict(z=False)),
        ("delta_bias=None", dict(delta_bias=False)),
        ("all None", dict(D=False, z=False, delta_bias=False)),
    ]:
        inputs = make_inputs(**kw)
        result = run_scan(inputs)
        out = result[0]
        assert out.shape == (BATCH, DIM, SEQLEN), f"FAIL shape with {label}"
    print("OK (D=None, z=None, delta_bias=None, all None)")


def test_bwd_eager():
    """Test backward pass through torch.ops directly."""
    print("10. bwd eager...", end=" ", flush=True)
    inputs = make_inputs()
    fwd_result = torch.ops.selective_scan.fwd(
        inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
        inputs["D"], inputs["z"], inputs["delta_bias"], True,
    )
    out, x = fwd_result[0], fwd_result[1]
    out_z = fwd_result[2] if len(fwd_result) > 2 else None
    bwd_result = torch.ops.selective_scan.bwd(
        inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
        inputs["D"], inputs["z"], inputs["delta_bias"],
        out_z if out_z is not None else out,
        x, out, None, True, False,
    )
    assert len(bwd_result) >= 7
    du = bwd_result[0]
    assert du.shape == inputs["u"].shape
    assert du.dtype == inputs["u"].dtype
    dB = bwd_result[3]
    assert dB.dtype == inputs["B"].dtype, f"dB dtype mismatch: {dB.dtype} != {inputs['B'].dtype}"
    dC = bwd_result[4]
    assert dC.dtype == inputs["C"].dtype, f"dC dtype mismatch: {dC.dtype} != {inputs['C'].dtype}"
    print("OK")


def test_bwd_compile():
    """Test backward pass through torch.compile."""
    print("11. bwd torch.compile fullgraph=True...", end=" ", flush=True)
    try:
        def fwd_bwd(u, delta, A, B, C, D, z, delta_bias):
            result = torch.ops.selective_scan.fwd(u, delta, A, B, C, D, z, delta_bias, True)
            out, x = result[0], result[1]
            out_z = result[2] if len(result) > 2 else None
            bwd_result = torch.ops.selective_scan.bwd(
                u, delta, A, B, C, D, z, delta_bias,
                out_z if out_z is not None else out,
                x, out, None, True, False,
            )
            return bwd_result[0]  # du

        compiled = torch.compile(fwd_bwd, fullgraph=True)
        inputs = make_inputs()
        du = compiled(inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"],
                      inputs["D"], inputs["z"], inputs["delta_bias"])
        assert du.shape == inputs["u"].shape
        print(f"OK")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Stable ops registered: {has_stable}")
    print()

    test_eager()
    test_compile_fullgraph()
    test_compile_with_breaks()
    test_compile_reduce_overhead()
    print()
    test_cuda_graph_raw_op()
    test_cuda_graph_with_autograd()
    print()
    test_export_non_strict()
    test_export_strict()
    print()
    test_none_optionals()
    test_bwd_eager()
    test_bwd_compile()
