# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_bwd_dqkwg` on Ascend NPU.

Run as a script:
    python fla/ops/common/test_chunk_bwd_dqkwg_perf_npu.py

NPU counterpart of `test_chunk_bwd_dqkwg_perf.py`. See that file for the
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

from fla.ops.common.chunk_o import chunk_bwd_dqkwg  # noqa: E402
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
    "/tmp/fla_chunk_bwd_dqkwg_perf_npu",
)


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
    g = torch.rand(B_DENSE, T_DENSE, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    g_gamma = torch.rand(HV, dtype=torch.float32, device=DEVICE).log()
    return {
        "q": q, "k": k, "v": v, "w": w, "do": do, "dv": dv,
        "h": h, "dh": dh, "g": g, "g_gamma": g_gamma,
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
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")
    print(f"H=HV={H} D={D} BT={BT} dtype={DTYPE}")

    dense = _build_dense_inputs()
    run_case("dense_no_gate_no_dw", dense, gate="none", with_dw=False)
    run_case("dense_with_g_no_dw", dense, gate="g", with_dw=False)
    run_case("dense_with_g_gamma_no_dw", dense, gate="g_gamma", with_dw=False)
    run_case("dense_with_g_with_dw", dense, gate="g", with_dw=True)


if __name__ == "__main__":
    main()
