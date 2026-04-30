# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Performance test for `chunk_gated_delta_rule_bwd_dhu` on Ascend NPU.

Run as a script:
    python fla/ops/common/test_chunk_gated_delta_rule_bwd_dhu_perf_npu.py

NPU counterpart of `test_chunk_gated_delta_rule_bwd_dhu_perf.py`. See that file
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

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu  # noqa: E402
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
    "/tmp/fla_chunk_gated_delta_rule_bwd_dhu_perf_npu",
)


def _build_inputs(B: int, T: int, N: int, cu_seqlens: torch.Tensor | None = None) -> dict:
    torch.manual_seed(42)
    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE), p=2, dim=-1).to(DTYPE)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE), p=2, dim=-1).to(DTYPE)
    w = (torch.randn(B, T, HV, D, dtype=torch.float32, device=DEVICE) * 0.1).to(DTYPE)
    do = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    dv = torch.randn(B, T, HV, D, dtype=DTYPE, device=DEVICE)
    g_raw = F.logsigmoid(torch.randn(B, T, HV, dtype=torch.float32, device=DEVICE))
    g = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=cu_seqlens)
    h0 = torch.randn(N, HV, D, D, dtype=torch.float32, device=DEVICE) * 0.1
    dht = torch.randn(N, HV, D, D, dtype=torch.float32, device=DEVICE) * 0.1
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
        trace_path=os.path.join(TRACE_PATH, label),
    )


def main() -> None:
    assert torch.npu.is_available(), "NPU required"
    name = torch.npu.get_device_name()
    print(f"NPU: {name}")
    print(f"H=HV={H} D={D} BT={BT} dtype={DTYPE}")

    dense = _build_inputs(B_DENSE, T_DENSE, N=B_DENSE)
    run_case("dense_no_g_no_h0_no_dht", dense, with_g=False, with_h0=False, with_dht=False, cu_seqlens=None)
    run_case("dense_with_g_no_h0_no_dht", dense, with_g=True, with_h0=False, with_dht=False, cu_seqlens=None)
    run_case("dense_with_g_with_h0_with_dht", dense, with_g=True, with_h0=True, with_dht=True, cu_seqlens=None)


if __name__ == "__main__":
    main()
