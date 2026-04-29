# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `prepare_wy_repr_bwd` on NVIDIA L20.

Run as a script:
    python fla/ops/gated_delta_rule/test_prepare_wy_repr_bwd_perf.py

Covers the only Triton kernel reached by `prepare_wy_repr_bwd`:
    `prepare_wy_repr_bwd_kernel`
across its meaningful constexpr branches:
    - IS_VARLEN: cu_seqlens is None vs. provided

Note: the kernel's USE_G=False branch leaves `b_dk` undefined and is never
exercised in production (the GDN backward in chunk.py always passes `g`),
so we only profile the USE_G=True path.

Shapes mirror the largest representative case from
`tests/ops/test_gated_delta.py::test_chunk` and ::test_chunk_varlen:
    non-varlen: B=2 T=2048 H=HV=4 D=128 BT=64
    varlen   : B=1 cu_seqlens=[0, 200, 512, 1200, 2048] H=HV=4 D=128 BT=64

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from fla.ops.perf_utils import profile_fn

H = 4
HV = 4
D = 128
BT = 64  # hardcoded in prepare_wy_repr_bwd

B_DENSE = 2
T_DENSE = 2048

CU_SEQLENS_LIST = [0, 200, 512, 1200, 2048]

DTYPE = torch.bfloat16
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_prepare_wy_repr_bwd_perf.json",
)


def _build_inputs(B: int, T: int) -> dict:
    torch.manual_seed(42)
    k = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    beta = torch.rand(B, T, HV, dtype=DTYPE, device=DEVICE).sigmoid()
    A = torch.randn(B, T, HV, BT, dtype=DTYPE, device=DEVICE) * 0.02
    dw = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    du = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    # g must be fp32: kernels load it without an explicit cast and feed it to
    # tl.exp/exp2, which reject bf16. In production, g is the fp32 output of
    # chunk_local_cumsum (output_dtype=torch.float).
    g = torch.rand(B, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    return {"k": k, "v": v, "beta": beta, "A": A, "dw": dw, "du": du, "g": g}


def run_case(label: str, base: dict, *, cu_seqlens: torch.Tensor | None) -> None:
    print(f"\n========== {label} ==========")
    kwargs = {
        "k": base["k"],
        "v": base["v"],
        "beta": base["beta"],
        "A": base["A"],
        "dw": base["dw"],
        "du": base["du"],
        "g": base["g"],
    }
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens

    dk, dv, db, dg = prepare_wy_repr_bwd(**kwargs)
    assert torch.isfinite(dk).all() and torch.isfinite(dv).all() and torch.isfinite(db).all(), "non-finite outputs"
    assert dg is not None and torch.isfinite(dg).all(), "dg non-finite"
    profile_fn(
        prepare_wy_repr_bwd,
        **kwargs,
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
    print(f"H=HV={H} D={D} BT={BT} dtype={DTYPE}")

    dense = _build_inputs(B_DENSE, T_DENSE)
    run_case("dense", dense, cu_seqlens=None)

    # T = CU_SEQLENS_LIST[-1]
    # cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)
    # varlen = _build_inputs(1, T)
    # run_case("varlen", varlen, cu_seqlens=cu_seqlens)


if __name__ == "__main__":
    main()
