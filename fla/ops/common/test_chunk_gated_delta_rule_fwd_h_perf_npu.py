# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_gated_delta_rule_fwd_h` on Ascend NPU.

Run as a script:
    python fla/ops/common/test_chunk_gated_delta_rule_fwd_h_perf_npu.py

NPU counterpart of `test_chunk_gated_delta_rule_fwd_h_perf.py`. See that file
for the covered Triton kernel and shape rationale; this script differs only in:
- `import torch_npu` to register `torch.npu`,
- `FLA_PROFILER_BACKEND=npu` so `profile_fn` routes to `torch_npu.profiler`,
- `DEVICE = "npu"`,
- `trace_path` is treated as a tensorboard trace directory.
"""

import os

os.environ.setdefault("FLA_PROFILER_BACKEND", "npu")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: E402, F401  (registers torch.npu)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from fla.ops.perf_utils import profile_fn  # noqa: E402
from fla.ops.utils import chunk_local_cumsum  # noqa: E402

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
    "/tmp/fla_chunk_gated_delta_rule_fwd_h_perf_npu",
)


def _build_dense_inputs() -> dict:
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B_DENSE, T_DENSE, H, D, dtype=torch.float32, device=DEVICE), p=2, dim=-1).to(DTYPE)
    w = (torch.randn(B_DENSE, T_DENSE, HV, D, dtype=torch.float32, device=DEVICE) * 0.1).to(DTYPE)
    u = torch.randn(B_DENSE, T_DENSE, HV, D, dtype=DTYPE, device=DEVICE)
    g_raw = F.logsigmoid(torch.randn(B_DENSE, T_DENSE, HV, dtype=torch.float32, device=DEVICE))
    g = chunk_local_cumsum(g_raw, chunk_size=BT)
    h0 = torch.randn(B_DENSE, HV, D, D, dtype=torch.float32, device=DEVICE) * 0.1
    return {"k": k, "w": w, "u": u, "g": g, "h0": h0}


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
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")
    print(f"H=HV={H} D={D} BT={BT} dtype={DTYPE}")

    dense = _build_dense_inputs()
    run_case("dense_no_g_no_h0", dense, with_g=False, with_h0=False, store_final=False)
    run_case("dense_with_g_no_h0", dense, with_g=True, with_h0=False, store_final=False)
    run_case("dense_with_g_with_h0_store_ht", dense, with_g=True, with_h0=True, store_final=True)


if __name__ == "__main__":
    main()
