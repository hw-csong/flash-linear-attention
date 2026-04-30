# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `recompute_w_u_fwd` on Ascend NPU.

Run as a script:
    python fla/ops/gated_delta_rule/test_recompute_w_u_fwd_perf_npu.py

NPU counterpart of `test_recompute_w_u_fwd_perf.py`. See that file for the
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

from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd  # noqa: E402
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
    "/tmp/fla_recompute_w_u_fwd_perf_npu",
)


def _build_dense_inputs() -> dict:
    torch.manual_seed(42)
    k = torch.randn(B_DENSE, T_DENSE, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    beta = torch.rand(B_DENSE, T_DENSE, HV, dtype=DTYPE, device=DEVICE).sigmoid()
    A = torch.zeros(B_DENSE, T_DENSE, HV, BT, dtype=DTYPE, device=DEVICE)
    A.normal_(mean=0.0, std=0.02)
    g = torch.rand(B_DENSE, T_DENSE, HV, dtype=torch.float32, device=DEVICE).log().cumsum(dim=1)
    return {"k": k, "v": v, "beta": beta, "A": A, "g": g}


def run_case(label: str, kwargs: dict) -> None:
    print(f"\n========== {label} ==========")
    w, u = recompute_w_u_fwd(**kwargs)
    assert torch.isfinite(w).all() and torch.isfinite(u).all(), "non-finite outputs"
    profile_fn(
        recompute_w_u_fwd,
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
    run_case("dense_no_g", {k: v for k, v in dense.items() if k != "g"})
    run_case("dense_with_g", dense)


if __name__ == "__main__":
    main()
