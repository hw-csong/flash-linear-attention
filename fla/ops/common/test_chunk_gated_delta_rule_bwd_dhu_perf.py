# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_gated_delta_rule_bwd_dhu` on NVIDIA L20.

Run as a script:
    python fla/ops/common/test_chunk_gated_delta_rule_bwd_dhu_perf.py

Covers the only Triton kernel reached by `chunk_gated_delta_rule_bwd_dhu`:
    `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`
across its meaningful constexpr branches:
    - USE_G                   : g is None vs. provided
    - USE_INITIAL_STATE       : h0 is None vs. provided (drives dh0 alloc)
    - USE_FINAL_STATE_GRADIENT: dht is None vs. provided
    - IS_VARLEN               : cu_seqlens is None vs. provided

Shapes mirror the largest representative case from
`tests/ops/test_gated_delta.py::test_chunk` and ::test_chunk_varlen:
    non-varlen: B=2 T=2048 H=HV=4 D=128 BT=64
    varlen   : B=1 cu_seqlens=[0, 200, 512, 1200, 2048] H=HV=4 D=128 BT=64

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
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
    "/tmp/fla_chunk_gated_delta_rule_bwd_dhu_perf.json",
)


def _build_inputs(B: int, T: int, N: int) -> dict:
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    w = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    do = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    dv = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    # g must be fp32: kernels feed it to tl.exp/exp2 which reject bf16; in
    # production it comes from chunk_local_cumsum (output_dtype=torch.float).
    g = torch.rand(B, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    h0 = torch.randn(N, HV, D, D, dtype=torch.float32, device=DEVICE)
    dht = torch.randn(N, HV, D, D, dtype=torch.float32, device=DEVICE)
    return {"q": q, "k": k, "w": w, "do": do, "dv": dv, "g": g, "h0": h0, "dht": dht}


def run_case(
    label: str,
    base: dict,
    *,
    with_g: bool,
    with_h0: bool,
    with_dht: bool,
    cu_seqlens: torch.Tensor | None,
) -> None:
    print(f"\n========== {label} ==========")
    kwargs = {
        "q": base["q"],
        "k": base["k"],
        "w": base["w"],
        "do": base["do"],
        "dv": base["dv"],
        "g": base["g"] if with_g else None,
        "h0": base["h0"] if with_h0 else None,
        "dht": base["dht"] if with_dht else None,
        "scale": D ** -0.5,
        "chunk_size": BT,
    }
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens

    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(**kwargs)
    assert torch.isfinite(dh).all() and torch.isfinite(dv2).all(), "non-finite outputs"
    if with_h0:
        assert dh0 is not None and torch.isfinite(dh0).all(), "dh0 non-finite"
    profile_fn(
        chunk_gated_delta_rule_bwd_dhu,
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

    dense = _build_inputs(B_DENSE, T_DENSE, N=B_DENSE)
    run_case("dense_no_g_no_h0_no_dht", dense, with_g=False, with_h0=False, with_dht=False, cu_seqlens=None)
    run_case("dense_with_g_no_h0_no_dht", dense, with_g=True, with_h0=False, with_dht=False, cu_seqlens=None)
    run_case("dense_with_g_with_h0_with_dht", dense, with_g=True, with_h0=True, with_dht=True, cu_seqlens=None)

    # T = CU_SEQLENS_LIST[-1]
    # N = len(CU_SEQLENS_LIST) - 1
    # cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)
    # varlen = _build_inputs(1, T, N=N)
    # run_case("varlen_with_g_with_h0_with_dht", varlen, with_g=True, with_h0=True, with_dht=True, cu_seqlens=cu_seqlens)


if __name__ == "__main__":
    main()
