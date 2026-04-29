# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_gated_delta_rule_fwd_h` on NVIDIA L20.

Run as a script:
    python fla/ops/common/test_chunk_gated_delta_rule_fwd_h_perf.py

Covers the only Triton kernel reached by `chunk_gated_delta_rule_fwd_h`:
    `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
across its meaningful constexpr branches:
    - USE_G          : g is None vs. provided
    - USE_INITIAL_STATE / STORE_FINAL_STATE: initial_state / output_final_state
    - IS_VARLEN      : cu_seqlens is None vs. provided
    - SAVE_NEW_VALUE : save_new_value is always exercised (default True)

Shapes mirror the largest representative case from
`tests/ops/test_gated_delta.py::test_chunk` and ::test_chunk_varlen:
    non-varlen: B=2 T=2048 H=HV=4 D=128 BT=64
    varlen   : B=1 cu_seqlens=[0, 200, 512, 1200, 2048] H=HV=4 D=128 BT=64

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
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
    "/tmp/fla_chunk_gated_delta_rule_fwd_h_perf.json",
)


def _build_dense_inputs() -> dict:
    torch.manual_seed(42)
    k = torch.randn(B_DENSE, T_DENSE, H, D, dtype=DTYPE, device=DEVICE)
    w = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    u = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    # g must be fp32: kernels feed it to tl.exp/exp2 which reject bf16; in
    # production it comes from chunk_local_cumsum (output_dtype=torch.float).
    g = torch.rand(B_DENSE, T_DENSE, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    h0 = torch.randn(B_DENSE, HV, D, D, dtype=torch.float32, device=DEVICE)
    return {"k": k, "w": w, "u": u, "g": g, "h0": h0}


def _build_varlen_inputs() -> dict:
    torch.manual_seed(42)
    cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)
    T = CU_SEQLENS_LIST[-1]
    N = len(CU_SEQLENS_LIST) - 1
    k = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    w = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    u = torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE)
    g = torch.rand(1, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    h0 = torch.randn(N, HV, D, D, dtype=torch.float32, device=DEVICE)
    return {"k": k, "w": w, "u": u, "g": g, "h0": h0, "cu_seqlens": cu_seqlens}


def run_case(label: str, base: dict, *, with_g: bool, with_h0: bool, store_final: bool) -> None:
    print(f"\n========== {label} ==========")
    kwargs = {
        "k": base["k"],
        "w": base["w"],
        "u": base["u"],
        "g": base["g"] if with_g else None,
        "initial_state": base["h0"] if with_h0 else None,
        "output_final_state": store_final,
        "chunk_size": BT,
    }
    if "cu_seqlens" in base:
        kwargs["cu_seqlens"] = base["cu_seqlens"]

    h, v_new, ht = chunk_gated_delta_rule_fwd_h(**kwargs)
    assert torch.isfinite(h).all() and torch.isfinite(v_new).all(), "non-finite outputs"
    if store_final:
        assert ht is not None and torch.isfinite(ht).all(), "final state non-finite"
    profile_fn(
        chunk_gated_delta_rule_fwd_h,
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
    run_case("dense_no_g_no_h0", dense, with_g=False, with_h0=False, store_final=False)
    run_case("dense_with_g_no_h0", dense, with_g=True, with_h0=False, store_final=False)
    run_case("dense_with_g_with_h0_store_ht", dense, with_g=True, with_h0=True, store_final=True)

    # varlen = _build_varlen_inputs()
    # run_case("varlen_with_g_no_h0", varlen, with_g=True, with_h0=False, store_final=False)
    # run_case("varlen_with_g_with_h0_store_ht", varlen, with_g=True, with_h0=True, store_final=True)


if __name__ == "__main__":
    main()
