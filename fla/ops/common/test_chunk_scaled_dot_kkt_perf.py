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

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.perf_utils import profile_fn

H = 4
D = 128
CHUNK_SIZE = 32
CU_SEQLENS_LIST = [0, 200, 512, 1200, 2048]

DTYPE = torch.bfloat16
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

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
    ref_ng = ref_chunk_scaled_dot_kkt_varlen(k, beta, None, cu_seqlens, CHUNK_SIZE)
    tri_ng = chunk_scaled_dot_kkt_fwd(
        k=k, g=None, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
    ).float()
    torch.testing.assert_close(tri_ng, ref_ng, rtol=5e-2, atol=5e-2)
    profile_fn(
        chunk_scaled_dot_kkt_fwd,
        k=k, g=None, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
        label="no_gate", warmup=WARMUP, iters=ITERS,
        trace_path=TRACE_PATH.replace(".json", ".no_gate.json"),
    )

    print("\n--- gated (USE_G=True) ---")
    ref_g = ref_chunk_scaled_dot_kkt_varlen(k, beta, g, cu_seqlens, CHUNK_SIZE)
    tri_g = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
    ).float()
    torch.testing.assert_close(tri_g, ref_g, rtol=5e-2, atol=5e-2)
    profile_fn(
        chunk_scaled_dot_kkt_fwd,
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE,
        label="gated", warmup=WARMUP, iters=ITERS,
        trace_path=TRACE_PATH.replace(".json", ".gated.json"),
    )


if __name__ == "__main__":
    main()
