# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_scaled_dot_kkt_fwd` on NVIDIA L20.

Run as a script:
    python fla/ops/common/test_chunk_scaled_dot_kkt_perf.py

Shape parameters mirror the last (largest) case from
`tests/ops/test_solve_tril.py::test_solve_tril_varlen`:
    H=4, D=128, chunk_size=32, cu_seqlens=[0, 200, 512, 1200, 2048]
The varlen path is used (B=1, bf16). Both `g=None` and `g != None` branches are
exercised.

Only depends on `fla.ops.common.chunk_scaled_dot_kkt`; the reference
implementation is inlined.
"""

import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

H = 4
D = 128
CHUNK_SIZE = 32
CU_SEQLENS_LIST = [0, 200, 512, 1200, 2048]

DTYPE = torch.bfloat16
DEVICE = "cuda"

WARMUP = 20
ITERS = 100

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_chunk_scaled_dot_kkt_perf.json",
)


def _ref_chunk_scaled_dot_kkt_dense(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    """Per-chunk strict-lower-triangular `beta * exp(g_diff) * (K @ K^T)`."""
    B_, T_, H_, _ = k.shape
    HV_ = beta.shape[2]
    repeat = HV_ // H_
    k_exp = k.repeat_interleave(repeat, dim=2) if repeat > 1 else k

    A = torch.zeros(B_, T_, HV_, chunk_size, dtype=torch.float32, device=k.device)
    for t0 in range(0, T_, chunk_size):
        t1 = min(t0 + chunk_size, T_)
        sz = t1 - t0
        kk = k_exp[:, t0:t1].float()
        bb = beta[:, t0:t1].float()
        kkt = torch.einsum('bihk,bjhk->bhij', kk, kk)
        if g is not None:
            gg = g[:, t0:t1].float().transpose(1, 2)
            kkt = kkt * torch.exp(gg.unsqueeze(-1) - gg.unsqueeze(-2))
        kkt = kkt * bb.transpose(1, 2).unsqueeze(-1)
        mask = torch.tril(
            torch.ones(sz, sz, dtype=torch.bool, device=k.device),
            diagonal=-1,
        )
        kkt = torch.where(mask, kkt, torch.zeros_like(kkt))
        A[:, t0:t1, :, :sz] = kkt.permute(0, 2, 1, 3)
    return A


def ref_chunk_scaled_dot_kkt_varlen(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Apply the dense reference to each sub-sequence carved out by `cu_seqlens`."""
    B_, T_, _, _ = k.shape
    HV_ = beta.shape[2]
    assert B_ == 1, "varlen requires batch size 1"
    A = torch.zeros(1, T_, HV_, chunk_size, dtype=torch.float32, device=k.device)
    bnds = cu_seqlens.tolist()
    for i in range(len(bnds) - 1):
        bos, eos = bnds[i], bnds[i + 1]
        if eos == bos:
            continue
        seq_g = g[:, bos:eos] if g is not None else None
        seq_A = _ref_chunk_scaled_dot_kkt_dense(
            k[:, bos:eos], beta[:, bos:eos], seq_g, chunk_size,
        )
        A[:, bos:eos] = seq_A
    return A


def n_chunks_total(cu_seqlens: list[int], chunk_size: int) -> int:
    return sum(
        ((cu_seqlens[i + 1] - cu_seqlens[i]) + chunk_size - 1) // chunk_size
        for i in range(len(cu_seqlens) - 1)
    )


def benchmark(
    label: str,
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
) -> float:
    """Warmup, then time `chunk_scaled_dot_kkt_fwd` with cuda events. Returns median ms."""
    ref = ref_chunk_scaled_dot_kkt_varlen(k, beta, g, cu_seqlens, CHUNK_SIZE)
    tri = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
    ).float()
    torch.testing.assert_close(tri, ref, rtol=5e-2, atol=5e-2)

    for _ in range(WARMUP):
        chunk_scaled_dot_kkt_fwd(
            k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
        )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        chunk_scaled_dot_kkt_fwd(
            k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
        )
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    NT = n_chunks_total(CU_SEQLENS_LIST, CHUNK_SIZE)
    flops = 2.0 * H * NT * (CHUNK_SIZE ** 2) * D
    tflops = flops / (median_ms * 1e-3) / 1e12
    print(f"  [{label}] median latency: {median_ms:.4f} ms  "
          f"effective MMA: {tflops:.2f} TFLOP/s  (L20 bf16 peak ≈ 119 TFLOP/s)")
    return median_ms


def profile_case(
    label: str,
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    trace_path: str,
) -> None:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(ITERS):
            with record_function(f"chunk_scaled_dot_kkt_fwd[{label}]"):
                chunk_scaled_dot_kkt_fwd(
                    k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
                )
        torch.cuda.synchronize()

    print(f"=== profile [{label}] (top by self_cuda_time_total) ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10,
        max_name_column_width=80,
    ))
    prof.export_chrome_trace(trace_path)
    print(f"chrome trace: {trace_path}")


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}")
    if "L20" not in name:
        print(f"  (target machine is L20; running on '{name}')")
    print(f"H={H} D={D} BT={CHUNK_SIZE} cu_seqlens={CU_SEQLENS_LIST} dtype={DTYPE}")

    T = CU_SEQLENS_LIST[-1]
    cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)

    torch.manual_seed(42)
    # Mirror test_solve_tril_varlen: k is L2-normalized along the last dim.
    k = torch.nn.functional.normalize(
        torch.randn((1, T, H, D), dtype=DTYPE, device=DEVICE), dim=-1,
    )
    beta = torch.randn((1, T, H), dtype=DTYPE, device=DEVICE).sigmoid()
    # Gate values are log-decays in (-inf, 0]; cumulative-summed before this kernel.
    g = torch.rand(1, T, H, dtype=DTYPE, device=DEVICE).log().cumsum(dim=1)

    print("\n--- no gate (USE_G=False) ---")
    benchmark("no_gate", k, beta, None, cu_seqlens)
    profile_case(
        "no_gate", k, beta, None, cu_seqlens,
        trace_path=TRACE_PATH.replace(".json", ".no_gate.json"),
    )

    print("\n--- gated (USE_G=True) ---")
    benchmark("gated", k, beta, g, cu_seqlens)
    profile_case(
        "gated", k, beta, g, cu_seqlens,
        trace_path=TRACE_PATH.replace(".json", ".gated.json"),
    )


if __name__ == "__main__":
    main()
