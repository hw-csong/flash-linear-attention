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

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch
import torch.nn.functional as F

from fla.ops.perf_utils import profile_fn
from fla.ops.utils.solve_tril import solve_tril

B = 4
T = 2048
H = 8
CHUNK_SIZE = 64        # must be one of {16, 32, 64} (asserted in solve_tril)
KDIM = 64              # inner dim used to construct a well-conditioned A
DTYPE = torch.float32
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

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

    tri = solve_tril(A_in)
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)

    profile_fn(
        solve_tril, A_in,
        label="solve_tril", warmup=WARMUP, iters=ITERS,
        trace_path=TRACE_PATH,
    )


if __name__ == "__main__":
    main()
