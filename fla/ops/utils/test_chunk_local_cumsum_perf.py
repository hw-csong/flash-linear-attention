# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_local_cumsum` on NVIDIA L20.

Run as a script:
    python fla/ops/utils/test_chunk_local_cumsum_perf.py

Covers both Triton kernels reached by `chunk_local_cumsum`:
    - `chunk_local_cumsum_scalar_kernel` : 3D input [B, T, H]
    - `chunk_local_cumsum_vector_kernel` : 4D input [B, T, H, D]
across the only meaningful constexpr branches that aren't varlen:
    - REVERSE   : reverse=False vs. reverse=True

Shapes mirror the largest representative dense case from
`tests/ops/test_utils.py::test_local_cumsum`:
    B=4 T=2048 H=8 C=128 D=1024 (dtype=fp32)

varlen paths are intentionally skipped per task scope.

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.perf_utils import profile_fn
from fla.ops.utils.cumsum import chunk_local_cumsum

B = 4
T = 2048
H = 8
C = 128
D = 1024

DTYPE = torch.float32
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_chunk_local_cumsum_perf.json",
)


def _ref_local_cumsum(s: torch.Tensor, chunk_size: int, reverse: bool) -> torch.Tensor:
    """Per-chunk cumsum along T (dim=1), with optional reverse."""
    out = torch.empty_like(s, dtype=torch.float32)
    T_ = s.shape[1]
    for i in range(0, T_, chunk_size):
        j = min(i + chunk_size, T_)
        chunk = s[:, i:j].float()
        if reverse:
            chunk = chunk.flip(1).cumsum(1).flip(1)
        else:
            chunk = chunk.cumsum(1)
        out[:, i:j] = chunk
    return out.to(s.dtype)


def run_case(label: str, s: torch.Tensor, *, reverse: bool) -> None:
    print(f"\n========== {label} ==========")
    ref = _ref_local_cumsum(s, C, reverse)
    tri = chunk_local_cumsum(s, chunk_size=C, reverse=reverse)
    torch.testing.assert_close(tri.to(ref.dtype), ref, rtol=1e-3, atol=1e-3)

    profile_fn(
        chunk_local_cumsum,
        s,
        chunk_size=C,
        reverse=reverse,
        label=label,
        warmup=WARMUP,
        iters=ITERS,
        trace_path=TRACE_PATH.replace(".json", f".{label}.json"),
    )


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}")
    if "L20" not in name:
        print(f"  (target machine is L20; running on '{name}')")
    print(f"B={B} T={T} H={H} C={C} D={D} dtype={DTYPE}")

    torch.manual_seed(42)

    # 3D input → chunk_local_cumsum_scalar_kernel
    s_scalar = torch.randn(B, T, H, dtype=DTYPE, device=DEVICE)
    run_case("scalar_forward", s_scalar, reverse=False)
    run_case("scalar_reverse", s_scalar, reverse=True)

    # 4D input → chunk_local_cumsum_vector_kernel
    s_vector = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    run_case("vector_forward", s_vector, reverse=False)
    run_case("vector_reverse", s_vector, reverse=True)


if __name__ == "__main__":
    main()
