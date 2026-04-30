# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_scaled_dot_kkt_fwd` on Ascend NPU.

Run as a script:
    python fla/ops/common/test_chunk_scaled_dot_kkt_perf_npu.py

NPU counterpart of `test_chunk_scaled_dot_kkt_perf.py`. See that file for the
covered Triton kernel and shape rationale; this script differs only in:
- `import torch_npu` to register `torch.npu`,
- `FLA_PROFILER_BACKEND=npu` so `profile_fn` routes to `torch_npu.profiler`,
- `DEVICE = "npu"`,
- `trace_path` is treated as a tensorboard trace directory.
"""

import os

os.environ.setdefault("FLA_PROFILER_BACKEND", "npu")

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401  (registers torch.npu)

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from fla.ops.perf_utils import profile_fn  # noqa: E402

BT = 64

# `tests/ops/test_delta.py::test_chunk` largest dense case.
NO_GATE = dict(B=4, T=2048, H=8, HV=8, K=64)
# `tests/ops/test_gated_delta_product.py::test_chunk` largest dense case.
GATED = dict(B=2, T=2048, H=8, HV=8, K=128)

DTYPE = torch.float16
DEVICE = "npu"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_chunk_scaled_dot_kkt_perf_npu",
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
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")

    run_case("no_gate", NO_GATE, with_g=False)
    run_case("gated", GATED, with_g=True)


if __name__ == "__main__":
    main()
