# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_bwd_dv_local` on Ascend NPU.

Run as a script:
    python fla/ops/common/test_chunk_bwd_dv_local_perf_npu.py

NPU counterpart of `test_chunk_bwd_dv_local_perf.py`. See that file for the
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

from fla.ops.common.chunk_o import chunk_bwd_dv_local  # noqa: E402
from fla.ops.perf_utils import profile_fn  # noqa: E402

H = 4
HV = 4
D = 128
BT = 64

B_DENSE = 2
T_DENSE = 2048

DTYPE = torch.bfloat16
DEVICE = "npu"

WARMUP = 5
ITERS = 20

TRACE_PATH = os.environ.get(
    "FLA_PERF_TRACE",
    "/tmp/fla_chunk_bwd_dv_local_perf_npu",
)


def _build_inputs(B: int, T: int) -> dict:
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    do = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    g = torch.rand(B, T, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    g_gamma = torch.rand(HV, dtype=torch.float32, device=DEVICE).log()
    A = torch.randn(B, T, HV, BT, dtype=DTYPE, device=DEVICE) * 0.02
    return {"q": q, "k": k, "do": do, "g": g, "g_gamma": g_gamma, "A": A}


def run_case(label: str, base: dict, *, gate: str, with_A: bool, cu_seqlens: torch.Tensor | None) -> None:
    print(f"\n========== {label} ==========")
    kwargs = {
        "q": base["q"],
        "k": base["k"],
        "do": base["do"],
        "g": base["g"] if gate == "g" else None,
        "g_gamma": base["g_gamma"] if gate == "g_gamma" else None,
        "A": base["A"] if with_A else None,
        "scale": D ** -0.5,
        "chunk_size": BT,
    }
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens

    dv = chunk_bwd_dv_local(**kwargs)
    assert torch.isfinite(dv).all(), "non-finite outputs"
    profile_fn(
        chunk_bwd_dv_local,
        **kwargs,
        label=label,
        warmup=WARMUP,
        iters=ITERS,
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")
    print(f"H=HV={H} D={D} BT={BT} dtype={DTYPE}")

    dense = _build_inputs(B_DENSE, T_DENSE)
    run_case("dense_no_gate", dense, gate="none", with_A=False, cu_seqlens=None)
    run_case("dense_with_g", dense, gate="g", with_A=False, cu_seqlens=None)
    run_case("dense_with_g_gamma", dense, gate="g_gamma", with_A=False, cu_seqlens=None)
    run_case("dense_with_A", dense, gate="none", with_A=True, cu_seqlens=None)


if __name__ == "__main__":
    main()
