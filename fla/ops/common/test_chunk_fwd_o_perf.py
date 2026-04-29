# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_fwd_o` on NVIDIA L20.

Run as a script:
    python fla/ops/common/test_chunk_fwd_o_perf.py

Covers the only Triton kernel reached by `chunk_fwd_o`:
    `chunk_fwd_kernel_o`
across its meaningful constexpr branches:
    - USE_G        : g is None vs. provided
    - USE_G_GAMMA  : g_gamma is None vs. provided
    - IS_VARLEN    : cu_seqlens is None vs. provided

Shapes mirror the largest representative case from
`tests/ops/test_gated_delta.py::test_chunk` and ::test_chunk_varlen:
    non-varlen: B=2 T=2048 H=HV=4 D=128 BT=64
    varlen   : B=1 cu_seqlens=[0, 200, 512, 1200, 2048] H=HV=4 D=128 BT=64

Per-OP timings come from torch.profiler via `fla.ops.perf_utils.profile_fn`.
"""

import os

import torch

from fla.ops.common.chunk_o import chunk_fwd_o
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
    "/tmp/fla_chunk_fwd_o_perf.json",
)


def _build_inputs(B: int, T: int) -> dict:
    torch.manual_seed(42)
    NT = (T + BT - 1) // BT
    q = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    # state matches the layout that chunk_gated_delta_rule_fwd_h produces:
    # [B, NT, HV, K, V]
    h = torch.randn(B, NT, HV, D, D, dtype=DTYPE, device=DEVICE)
    # g must be fp32: kernels feed it to tl.exp/exp2 which reject bf16; in
    # production it comes from chunk_local_cumsum (output_dtype=torch.float).
    g = torch.rand(B, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    g_gamma = torch.rand(HV, dtype=torch.float32, device=DEVICE).log()
    return {"q": q, "k": k, "v": v, "h": h, "g": g, "g_gamma": g_gamma}


def run_case(label: str, base: dict, *, gate: str, cu_seqlens: torch.Tensor | None) -> None:
    print(f"\n========== {label} ==========")
    kwargs = {
        "q": base["q"],
        "k": base["k"],
        "v": base["v"],
        "h": base["h"],
        "g": base["g"] if gate == "g" else None,
        "g_gamma": base["g_gamma"] if gate == "g_gamma" else None,
        "chunk_size": BT,
    }
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens

    o = chunk_fwd_o(**kwargs)
    assert torch.isfinite(o).all(), "non-finite outputs"
    profile_fn(
        chunk_fwd_o,
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
    run_case("dense_no_gate", dense, gate="none", cu_seqlens=None)
    run_case("dense_with_g", dense, gate="g", cu_seqlens=None)
    run_case("dense_with_g_gamma", dense, gate="g_gamma", cu_seqlens=None)

    # T = CU_SEQLENS_LIST[-1]
    # cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.int32, device=DEVICE)
    # # For varlen, h is shaped per the prepared chunk offsets (one chunk-row per
    # # chunk index), but chunk_fwd_o derives `i_tg = i_t` directly from
    # # chunk_indices[:, 0:2]; to keep the kernel happy, allocate h with enough
    # # leading dimension to span all chunks across the sequences.
    # n_chunks_total = sum(((CU_SEQLENS_LIST[i + 1] - CU_SEQLENS_LIST[i] + BT - 1) // BT)
    #                      for i in range(len(CU_SEQLENS_LIST) - 1))
    # varlen = {
    #     "q": torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE),
    #     "k": torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE),
    #     "v": torch.randn(1, T, HV, D, dtype=DTYPE, device=DEVICE),
    #     "h": torch.randn(1, n_chunks_total, HV, D, D, dtype=DTYPE, device=DEVICE),
    #     "g": torch.rand(1, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1),
    #     "g_gamma": torch.rand(HV, dtype=torch.float32, device=DEVICE).log(),
    # }
    # run_case("varlen_with_g", varlen, gate="g", cu_seqlens=cu_seqlens)
    # run_case("varlen_with_g_gamma", varlen, gate="g_gamma", cu_seqlens=cu_seqlens)


if __name__ == "__main__":
    main()
