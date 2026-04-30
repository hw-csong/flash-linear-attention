# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_local_cumsum` on Ascend NPU.

Run as a script:
    python fla/ops/utils/test_chunk_local_cumsum_perf_npu.py

NPU counterpart of `test_chunk_local_cumsum_perf.py`. See that file for the
covered Triton kernels and shape rationale; this script differs only in:
- `import torch_npu` to register `torch.npu`,
- `FLA_PROFILER_BACKEND=npu` so `profile_fn` routes to `torch_npu.profiler`,
- `DEVICE = "npu"`,
- `trace_path` is treated as a tensorboard trace directory.
"""

import os

os.environ.setdefault("FLA_PROFILER_BACKEND", "npu")

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401  (registers torch.npu)

from fla.ops.perf_utils import profile_fn  # noqa: E402
from fla.ops.utils.cumsum import chunk_local_cumsum  # noqa: E402

B = 4
T = 2048
H = 8
C = 128
D = 1024

DTYPE = torch.float32
DEVICE = "npu"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_chunk_local_cumsum_perf_npu",
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
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")
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
