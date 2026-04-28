# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_local_cumsum` on NVIDIA L20.

Run as a script:
    python fla/ops/utils/test_local_cumsum_perf.py

Mirrors the largest non-varlen shape from `tests/ops/test_utils.py::test_local_cumsum`
(B=4, T=2048, H=8, C=128, D=1024, dtype=float32) and exercises both the scalar
(3D input) and vector (4D input) Triton paths.

Only depends on `fla.ops.utils.cumsum` — the reference cumsum is inlined.
"""

import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from fla.ops.utils.cumsum import chunk_local_cumsum

B, T, H, C, D = 4, 2048, 8, 128, 1024
DTYPE = torch.float32
DEVICE = "cuda"

WARMUP = 20
ITERS = 100

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_local_cumsum_perf.json",
)


def ref_local_cumsum(s: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Per-chunk cumsum along the time dim — matches `test_local_cumsum`."""
    return torch.cat(
        [s[:, i:i + chunk_size].float().cumsum(1) for i in range(0, s.shape[1], chunk_size)],
        dim=1,
    )


def benchmark(name: str, s: torch.Tensor, chunk_size: int) -> float:
    """Warm up, then time `chunk_local_cumsum` with cuda events. Returns median ms."""
    # Sanity check vs reference (loose tol because Triton accumulates in fp32).
    ref = ref_local_cumsum(s, chunk_size=chunk_size)
    tri = chunk_local_cumsum(s, chunk_size=chunk_size).to(ref.dtype)
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)

    # Warmup covers Triton autotune + JIT compile.
    for _ in range(WARMUP):
        chunk_local_cumsum(s, chunk_size=chunk_size)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        chunk_local_cumsum(s, chunk_size=chunk_size)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    nbytes = 2 * s.numel() * s.element_size()  # read + write
    bw_gbs = nbytes / (median_ms * 1e-3) / 1e9
    print(f"  [{name}] shape={tuple(s.shape)} dtype={s.dtype}")
    print(f"  [{name}] median latency: {median_ms:.4f} ms")
    print(f"  [{name}] effective bw : {bw_gbs:.1f} GB/s  (L20 peak ≈ 864 GB/s)")
    return median_ms


def profile_case(label: str, s: torch.Tensor, chunk_size: int, trace_path: str) -> None:
    """Run torch.profiler over `ITERS` calls and dump a Chrome trace."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(ITERS):
            with record_function(f"chunk_local_cumsum[{label}]"):
                chunk_local_cumsum(s, chunk_size=chunk_size)
        torch.cuda.synchronize()

    print()
    print(f"=== profile [{label}] ===")
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
    print(f"B={B} T={T} H={H} C={C} D={D} dtype={DTYPE}")

    torch.manual_seed(42)

    # 3D scalar-gate path: (B, T, H) — exercises chunk_local_cumsum_scalar_kernel
    print("\n--- scalar (3D) ---")
    s3 = torch.randn(B, T, H, dtype=DTYPE, device=DEVICE)
    benchmark("scalar", s3, chunk_size=C)
    profile_case("scalar", s3, chunk_size=C, trace_path=TRACE_PATH.replace(".json", ".scalar.json"))

    # 4D vector-gate path: (B, T, H, D) — exercises chunk_local_cumsum_vector_kernel
    print("\n--- vector (4D) ---")
    s4 = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    benchmark("vector", s4, chunk_size=C)
    profile_case("vector", s4, chunk_size=C, trace_path=TRACE_PATH.replace(".json", ".vector.json"))


if __name__ == "__main__":
    main()
