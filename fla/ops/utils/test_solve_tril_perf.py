# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `solve_tril` on NVIDIA L20.

Run as a script:
    python fla/ops/utils/test_solve_tril_perf.py

Shape parameters mirror the last (largest) case from
`tests/ops/test_solve_tril.py::test_solve_tril`:
    B=4, T=2048, H=8, chunk_size=64, dtype=float32
The dense (non-varlen) path is exercised; chunk_size=64 routes to
`merge_16x16_to_64x64_inverse_kernel`.

Only depends on `fla.ops.utils.solve_tril`; the reference uses `torch.inverse`.
"""

import os

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from fla.ops.utils.solve_tril import solve_tril

B = 4
T = 2048
H = 8
CHUNK_SIZE = 64        # must be one of {16, 32, 64} (asserted in solve_tril)
KDIM = 64              # inner dim used to construct a well-conditioned A
DTYPE = torch.float32
DEVICE = "cuda"

WARMUP = 20
ITERS = 100

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_solve_tril_perf.json",
)


def build_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (A_for_solve_tril, ref_inverse) following test_solve_tril's setup.

    `solve_tril` expects A of shape [B, T, H, BT], strictly lower-triangular.
    We construct it from L2-normalized K so that I + A is well-conditioned.
    Returns:
        A_in:  [B, T, H, BT]              -- input to solve_tril
        ref:   [B, T, H, BT]              -- (I + A)^-1 reference
    """
    # do not randomly initialize A otherwise the inverse is not stable
    k = F.normalize(
        torch.randn((B, H, T, KDIM), dtype=DTYPE, device=DEVICE),
        dim=-1,
    )
    padding_size = (CHUNK_SIZE - T % CHUNK_SIZE) % CHUNK_SIZE
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, CHUNK_SIZE, KDIM)
    # [B, H, num_chunks, BT, BT], strict lower-triangular per chunk
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)

    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)[None, None, None, ...]
    ref = torch.inverse(A + eye)
    # match the layout expected by solve_tril: [B, T, H, BT]
    ref = ref.reshape(B, H, -1, CHUNK_SIZE)[:, :, :T, :].transpose(1, 2).contiguous()

    A_in = A.reshape(B, H, -1, CHUNK_SIZE)[:, :, :T, :].transpose(1, 2).contiguous()
    return A_in, ref


def benchmark(label: str, A_in: torch.Tensor, ref: torch.Tensor) -> float:
    """Warmup, then time `solve_tril` with cuda events. Returns median ms."""
    tri = solve_tril(A_in)
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)

    for _ in range(WARMUP):
        solve_tril(A_in)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        solve_tril(A_in)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    # Each chunk inverts a BT x BT lower-triangular matrix; cost dominated by
    # the 16x16 -> 64x64 merge mat-muls. Report effective throughput as the
    # number of BT x BT inversions per second.
    n_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    inv_per_s = (B * H * n_chunks) / (median_ms * 1e-3)
    print(f"  [{label}] median latency: {median_ms:.4f} ms  "
          f"throughput: {inv_per_s / 1e6:.2f} M chunk-inverses/s")
    return median_ms


def profile_case(label: str, A_in: torch.Tensor, trace_path: str) -> None:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(ITERS):
            with record_function(f"solve_tril[{label}]"):
                solve_tril(A_in)
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
    print(f"B={B} T={T} H={H} BT={CHUNK_SIZE} dtype={DTYPE}")

    torch.manual_seed(42)
    A_in, ref = build_inputs()
    print(f"A_in shape: {tuple(A_in.shape)}  dtype: {A_in.dtype}")

    benchmark("solve_tril", A_in, ref)
    profile_case("solve_tril", A_in, trace_path=TRACE_PATH)


if __name__ == "__main__":
    main()
