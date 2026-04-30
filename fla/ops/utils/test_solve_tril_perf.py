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

Covers every Triton kernel reached by `solve_tril`, dispatched by `BT`:
    - `solve_tril_16x16_kernel`              (BT=16)
    - `merge_16x16_to_32x32_inverse_kernel`  (BT=32)
    - `merge_16x16_to_64x64_inverse_kernel`  (BT=64)

For each `BT` we use the largest case at that `chunk_size` from
`tests/ops/test_solve_tril.py::test_solve_tril`:
    BT=16: B=1 T=63   H=1
    BT=32: B=2 T=500  H=4
    BT=64: B=4 T=2048 H=8
    DK=64 (matches the reference test) and dtype=fp32.

The input `A` is constructed exactly like the reference test: the strict
lower-triangular part of a per-chunk Gram matrix of an L2-normalised tensor,
keeping the inverse well-conditioned.

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch
import torch.nn.functional as F

from fla.ops.perf_utils import profile_fn
from fla.ops.utils.solve_tril import solve_tril

# Per-BT shape picked as the largest case at that chunk_size from
# tests/ops/test_solve_tril.py::test_solve_tril.
SHAPES_PER_BT: dict[int, dict[str, int]] = {
    16: dict(B=1, T=63,   H=1),
    32: dict(B=2, T=500,  H=4),
    64: dict(B=4, T=2048, H=8),
}
DK = 64

DTYPE = torch.float32
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_solve_tril_perf.json",
)


def _build_A(chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Construct (A, ref_inverse, shape) following `tests/ops/test_solve_tril.py`."""
    shape = SHAPES_PER_BT[chunk_size]
    B, T, H = shape["B"], shape["T"], shape["H"]
    torch.manual_seed(42)
    k = F.normalize(
        torch.randn((B, H, T, DK), dtype=DTYPE, device=DEVICE),
        dim=-1,
    )
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, DK)
    A_full = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)

    eye = torch.eye(chunk_size, device=DEVICE, dtype=DTYPE)[None, None, None, ...]
    ref = torch.inverse(A_full + eye)
    ref = ref.reshape(B, H, -1, chunk_size)[:, :, :T, :]

    A = A_full.reshape(B, H, -1, chunk_size)[:, :, :T, :].transpose(1, 2).contiguous()
    return A, ref, shape


def run_case(chunk_size: int) -> None:
    label = f"BT{chunk_size}"
    A, ref, shape = _build_A(chunk_size)
    print(f"\n========== {label} (B={shape['B']} T={shape['T']} H={shape['H']} DK={DK}) ==========")

    tri = solve_tril(A).transpose(1, 2)
    torch.testing.assert_close(tri.to(ref.dtype), ref, rtol=1e-3, atol=1e-3)

    profile_fn(
        solve_tril,
        A,
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
    print(f"DK={DK} dtype={DTYPE}")

    for chunk_size in (16, 32, 64):
        run_case(chunk_size)


if __name__ == "__main__":
    main()
