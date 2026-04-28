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

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.perf_utils import profile_fn
from fla.ops.utils.cumsum import chunk_local_cumsum

B, T, H, C, D = 4, 2048, 8, 128, 1024
DTYPE = torch.float32
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

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
    ref3 = ref_local_cumsum(s3, chunk_size=C)
    tri3 = chunk_local_cumsum(s3, chunk_size=C).to(ref3.dtype)
    torch.testing.assert_close(tri3, ref3, rtol=1e-3, atol=1e-3)
    profile_fn(
        chunk_local_cumsum, s3, chunk_size=C,
        label="scalar", warmup=WARMUP, iters=ITERS,
        trace_path=TRACE_PATH.replace(".json", ".scalar.json"),
    )

    # 4D vector-gate path: (B, T, H, D) — exercises chunk_local_cumsum_vector_kernel
    print("\n--- vector (4D) ---")
    s4 = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    ref4 = ref_local_cumsum(s4, chunk_size=C)
    tri4 = chunk_local_cumsum(s4, chunk_size=C).to(ref4.dtype)
    torch.testing.assert_close(tri4, ref4, rtol=1e-3, atol=1e-3)
    profile_fn(
        chunk_local_cumsum, s4, chunk_size=C,
        label="vector", warmup=WARMUP, iters=ITERS,
        trace_path=TRACE_PATH.replace(".json", ".vector.json"),
    )


if __name__ == "__main__":
    main()
