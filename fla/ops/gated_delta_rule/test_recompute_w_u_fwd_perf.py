# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `recompute_w_u_fwd` on NVIDIA L20.

Run as a script:
    python fla/ops/gated_delta_rule/test_recompute_w_u_fwd_perf.py

Covers the only Triton kernel reached by `recompute_w_u_fwd`:
    `recompute_w_u_fwd_kernel`
across its meaningful constexpr branches:
    - USE_G  : g is None vs. provided
    - IS_VARLEN: cu_seqlens is None vs. provided

Shapes mirror the largest representative case from
`tests/ops/test_gated_delta.py::test_chunk` and ::test_chunk_varlen:
    non-varlen: B=2 T=2048 H=HV=4 D=128 BT=64
    varlen   : B=1 cu_seqlens=[0, 200, 512, 1200, 2048] H=HV=4 D=128 BT=64

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.perf_utils import profile_fn

H = 4
HV = 4
D = 128
BT = 64

B_DENSE = 2
T_DENSE = 2048

CU_SEQLENS_LIST = [0, 200, 512, 1200, 2048]

DTYPE = torch.bfloat16
DEVICE = "cuda"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_recompute_w_u_fwd_perf.json",
)


def _build_dense_inputs() -> dict:
    torch.manual_seed(42)
    k = torch.randn(B_DENSE, T_DENSE, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    beta = torch.rand(B_DENSE, T_DENSE, HV, dtype=DTYPE, device=DEVICE).sigmoid()
    # A ~ (I + strict_lower)^{-1}; here we just need a numerically reasonable BT-block tensor.
    A = torch.zeros(B_DENSE, T_DENSE, HV, BT, dtype=DTYPE, device=DEVICE)
    A.normal_(mean=0.0, std=0.02)
    # g must be fp32: kernels feed it to tl.exp/exp2 which reject bf16; in
    # production it comes from chunk_local_cumsum (output_dtype=torch.float).
    g = torch.rand(B_DENSE, T_DENSE, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    return {"k": k, "v": v, "beta": beta, "A": A, "g": g}


def _build_varlen_inputs() -> dict:
    torch.manual_seed(42)
    cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)
    T = CU_SEQLENS_LIST[-1]
    k = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    beta = torch.rand(1, T, HV, dtype=DTYPE, device=DEVICE).sigmoid()
    A = torch.zeros(1, T, HV, BT, dtype=DTYPE, device=DEVICE)
    A.normal_(mean=0.0, std=0.02)
    g = torch.rand(1, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    return {"k": k, "v": v, "beta": beta, "A": A, "g": g, "cu_seqlens": cu_seqlens}


def run_case(label: str, kwargs: dict) -> None:
    print(f"\n========== {label} ==========")
    # Sanity check: kernel runs and writes valid (non-NaN) outputs.
    w, u = recompute_w_u_fwd(**kwargs)
    assert torch.isfinite(w).all() and torch.isfinite(u).all(), "non-finite outputs"
    profile_fn(
        recompute_w_u_fwd,
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

    dense = _build_dense_inputs()
    run_case("dense_no_g", {k: v for k, v in dense.items() if k != "g"})
    run_case("dense_with_g", dense)

    # varlen = _build_varlen_inputs()
    # no_g = {k: v for k, v in varlen.items() if k != "g"}
    # run_case("varlen_no_g", no_g)
    # run_case("varlen_with_g", varlen)


if __name__ == "__main__":
    main()
