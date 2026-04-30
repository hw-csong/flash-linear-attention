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

Covers the only Triton kernel reached by `chunk_scaled_dot_kkt_fwd`:
    `chunk_scaled_dot_kkt_fwd_kernel`
across the meaningful constexpr branches that aren't varlen:
    - USE_G : g is None vs. provided

Shapes mirror the largest representative dense cases from `tests/ops/`:
    - USE_G=False : `tests/ops/test_delta.py::test_chunk` largest case
        (B=4, T=2048, H=HV=8, D=64, dtype=fp16)
        — delta_rule's chunk path calls `chunk_scaled_dot_kkt_fwd` with g=None.
    - USE_G=True  : `tests/ops/test_gated_delta_product.py::test_chunk`
        largest case (B=2, T=2048, H=HV=8, D=128, dtype=fp16)
        — gated_delta_product's chunk path passes a non-None `g`.

varlen paths are intentionally skipped per task scope.

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.perf_utils import profile_fn

BT = 64

# `tests/ops/test_delta.py::test_chunk` largest dense case.
NO_GATE = dict(B=4, T=2048, H=8, HV=8, K=64)
# `tests/ops/test_gated_delta_product.py::test_chunk` largest dense case.
GATED = dict(B=2, T=2048, H=8, HV=8, K=128)

DTYPE = torch.float16
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_chunk_scaled_dot_kkt_perf.json",
)


def _ref_chunk_scaled_dot_kkt(
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


def run_case(label: str, shape: dict, *, with_g: bool) -> None:
    print(f"\n========== {label} ==========")
    B, T, H, HV, K = shape["B"], shape["T"], shape["H"], shape["HV"], shape["K"]
    print(f"B={B} T={T} H={H} HV={HV} K={K} BT={BT} dtype={DTYPE}")

    torch.manual_seed(42)
    # Mirror tests/ops/test_solve_tril.py: k is L2-normalized along the last dim.
    k = torch.nn.functional.normalize(
        torch.randn((B, T, H, K), dtype=DTYPE, device=DEVICE), dim=-1,
    )
    beta = torch.randn((B, T, HV), dtype=DTYPE, device=DEVICE).sigmoid()
    g = (
        torch.rand(B, T, HV, dtype=DTYPE, device=DEVICE).log().cumsum(dim=1)
        if with_g else None
    )

    ref = _ref_chunk_scaled_dot_kkt(k, beta, g, BT)
    tri = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, chunk_size=BT,
    ).float()
    torch.testing.assert_close(tri, ref, rtol=5e-2, atol=5e-2)

    profile_fn(
        chunk_scaled_dot_kkt_fwd,
        k=k, g=g, beta=beta, chunk_size=BT,
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

    run_case("no_gate", NO_GATE, with_g=False)
    run_case("gated", GATED, with_g=True)


if __name__ == "__main__":
    main()
