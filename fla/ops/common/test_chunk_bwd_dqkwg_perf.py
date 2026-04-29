# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_bwd_dqkwg` on NVIDIA L20.

Run as a script:
    python fla/ops/common/test_chunk_bwd_dqkwg_perf.py

Covers the only Triton kernel reached by `chunk_bwd_dqkwg`:
    `chunk_bwd_kernel_dqkwg`
across its meaningful constexpr branches:
    - USE_G        : g is None vs. provided
    - USE_G_GAMMA  : g_gamma is None vs. provided
    - USE_DW       : w (delta-rule) is None vs. provided (also gates dv usage)
    - IS_VARLEN    : cu_seqlens is None vs. provided

Note: the function raises on Hopper + Triton >= 3.4.0 with `g != None`. This is
fine on L20 (Ada).

Shapes mirror the largest representative case from
`tests/ops/test_gated_delta.py::test_chunk` and ::test_chunk_varlen:
    non-varlen: B=2 T=2048 H=HV=4 D=128 BT=64
    varlen   : B=1 cu_seqlens=[0, 200, 512, 1200, 2048] H=HV=4 D=128 BT=64

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.common.chunk_o import chunk_bwd_dqkwg
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
    "/tmp/fla_chunk_bwd_dqkwg_perf.json",
)


def _num_chunks_total(cu_seqlens: list[int]) -> int:
    return sum(((cu_seqlens[i + 1] - cu_seqlens[i] + BT - 1) // BT)
               for i in range(len(cu_seqlens) - 1))


def _build_dense_inputs() -> dict:
    torch.manual_seed(42)
    NT = (T_DENSE + BT - 1) // BT
    q = torch.randn(B_DENSE, T_DENSE, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B_DENSE, T_DENSE, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    w = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    do = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    dv = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    h = torch.randn(B_DENSE, NT, HV, D, D, dtype=DTYPE, device=DEVICE)
    dh = torch.randn(B_DENSE, NT, HV, D, D, dtype=DTYPE, device=DEVICE)
    # g must be fp32: kernels feed it to tl.exp/exp2 which reject bf16; in
    # production it comes from chunk_local_cumsum (output_dtype=torch.float).
    g = torch.rand(B_DENSE, T_DENSE, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    g_gamma = torch.rand(HV, dtype=torch.float32, device=DEVICE).log()
    return {
        "q": q, "k": k, "v": v, "w": w, "do": do, "dv": dv,
        "h": h, "dh": dh, "g": g, "g_gamma": g_gamma,
    }


def _build_varlen_inputs() -> dict:
    torch.manual_seed(42)
    cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)
    T = CU_SEQLENS_LIST[-1]
    NT = _num_chunks_total(CU_SEQLENS_LIST)
    q = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    w = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    do = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    dv = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    h = torch.randn(1, NT, HV, D, D, dtype=DTYPE, device=DEVICE)
    dh = torch.randn(1, NT, HV, D, D, dtype=DTYPE, device=DEVICE)
    g = torch.rand(1, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    g_gamma = torch.rand(HV, dtype=torch.float32, device=DEVICE).log()
    return {
        "q": q, "k": k, "v": v, "w": w, "do": do, "dv": dv,
        "h": h, "dh": dh, "g": g, "g_gamma": g_gamma,
        "cu_seqlens": cu_seqlens,
    }


def run_case(label: str, base: dict, *, gate: str, with_dw: bool) -> None:
    print(f"\n========== {label} ==========")
    kwargs = {
        "q": base["q"],
        "k": base["k"],
        "v": base["v"],
        "do": base["do"],
        "h": base["h"],
        "dh": base["dh"],
        "w": base["w"] if with_dw else None,
        "g": base["g"] if gate == "g" else None,
        "g_gamma": base["g_gamma"] if gate == "g_gamma" else None,
        "dv": base["dv"] if with_dw else None,
        "scale": D ** -0.5,
        "chunk_size": BT,
    }
    if "cu_seqlens" in base:
        kwargs["cu_seqlens"] = base["cu_seqlens"]

    dq, dk, dw, dg = chunk_bwd_dqkwg(**kwargs)
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all(), "non-finite outputs"
    if with_dw:
        assert dw is not None and torch.isfinite(dw).all(), "dw non-finite"
    if gate == "g":
        assert dg is not None and torch.isfinite(dg).all(), "dg non-finite"
    profile_fn(
        chunk_bwd_dqkwg,
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
    run_case("dense_no_gate_no_dw", dense, gate="none", with_dw=False)
    run_case("dense_with_g_no_dw", dense, gate="g", with_dw=False)
    run_case("dense_with_g_gamma_no_dw", dense, gate="g_gamma", with_dw=False)
    run_case("dense_with_g_with_dw", dense, gate="g", with_dw=True)

    # varlen = _build_varlen_inputs()
    # run_case("varlen_with_g_no_dw", varlen, gate="g", with_dw=False)
    # run_case("varlen_with_g_with_dw", varlen, gate="g", with_dw=True)


if __name__ == "__main__":
    main()
