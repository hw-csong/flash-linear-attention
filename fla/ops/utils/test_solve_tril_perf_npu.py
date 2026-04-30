# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `solve_tril` on Ascend NPU.

Run as a script:
    python fla/ops/utils/test_solve_tril_perf_npu.py

NPU counterpart of `test_solve_tril_perf.py`. See that file for the covered
Triton kernels (BT=16/32/64 dispatch) and shape rationale; this script differs
only in:
- `import torch_npu` to register `torch.npu`,
- `FLA_PROFILER_BACKEND=npu` so `profile_fn` routes to `torch_npu.profiler`,
- `DEVICE = "npu"`,
- `trace_path` is treated as a tensorboard trace directory.
"""

import os

os.environ.setdefault("FLA_PROFILER_BACKEND", "npu")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: E402, F401  (registers torch.npu)

from fla.ops.perf_utils import profile_fn  # noqa: E402
from fla.ops.utils.solve_tril import solve_tril  # noqa: E402

B = 4
T = 2048
H = 8
DK = 64

DTYPE = torch.float32
DEVICE = "npu"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_solve_tril_perf_npu",
)


def _build_A(chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct (A, ref_inverse) following `tests/ops/test_solve_tril.py`."""
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
    return A, ref


def run_case(chunk_size: int) -> None:
    label = f"BT{chunk_size}"
    print(f"\n========== {label} ==========")
    A, ref = _build_A(chunk_size)

    tri = solve_tril(A).transpose(1, 2)
    torch.testing.assert_close(tri.to(ref.dtype), ref, rtol=1e-3, atol=1e-3)

    profile_fn(
        solve_tril,
        A,
        label=label,
        warmup=WARMUP,
        iters=ITERS,
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")
    print(f"B={B} T={T} H={H} DK={DK} dtype={DTYPE}")

    for chunk_size in (16, 32, 64):
        run_case(chunk_size)


if __name__ == "__main__":
    main()
