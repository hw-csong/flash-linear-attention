# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Shared utility for the `test_*_perf.py` scripts under `fla/ops/**`.

Single entry point:
- `profile_fn(fn, *args, **kwargs)` — torch.profiler trace, per-OP timing
  table sorted by average self-device time, optional trace export.

Backend selection is controlled by the `FLA_PROFILER_BACKEND` environment
variable:
- unset / "cuda" (default): use `torch.profiler` with CPU + CUDA activities.
- "npu": use `torch_npu.profiler` with CPU + NPU activities.

External callers do not change between backends.

Per-script setup (input construction, correctness check, output formatting)
stays in the calling script.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

_BACKEND_ENV = "FLA_PROFILER_BACKEND"


def _resolve_backend() -> str:
    backend = os.environ.get(_BACKEND_ENV, "").strip().lower()
    if backend == "npu":
        return "npu"
    return "cuda"


def profile_fn(
    fn: Callable[..., Any],
    *args: Any,
    label: str | None = None,
    warmup: int = 5,
    iters: int = 20,
    trace_path: str | None = None,
    row_limit: int = 10,
    sort_by: str = "self_cuda_time_total",
    **kwargs: Any,
) -> Any:
    """
    Profile `fn(*args, **kwargs)` and print per-OP timings.

    Reports per-operator (kernel) time as observed by the profiler — no
    cuda/npu-event wall-clock timing.

    Backend is selected by the `FLA_PROFILER_BACKEND` environment variable:
    unset/"cuda" routes to `torch.profiler`, "npu" routes to
    `torch_npu.profiler`.

    Args:
        fn: callable to profile; the work must hit the GPU/NPU.
        *args, **kwargs: forwarded verbatim to `fn` on every call.
        label: tag printed with the result and used in `record_function`.
               Defaults to `fn.__name__`.
        warmup: warmup iterations run *outside* the profiler window. Covers
                Triton autotune + JIT compile so they don't pollute the report.
        iters: iterations recorded inside the profiler window.
        trace_path: if set, export a Chrome trace (CUDA) or write a
                tensorboard trace dir (NPU) to this path.
        row_limit: how many top rows of the table to print.
        sort_by: column to sort the table by. Defaults to
                 `self_cuda_time_total` — totals across `iters`, which (since
                 every key is hit `iters` times) ranks identically to per-call
                 average time, but avoids a torch.profiler attribute-access bug
                 on `self_cuda_time` in recent torch versions
                 (FunctionEventAvg.self_device_time AttributeError). The table
                 still displays the "CUDA avg" column.
                 Other valid choices: 'cuda_time_total', 'self_cpu_time_total',
                 'cpu_time_total', 'count'. On the NPU backend, "cuda" in
                 sort_by is auto-mapped to "npu".

    Returns:
        The profiler's `key_averages()` event list, for further inspection.
    """
    if _resolve_backend() == "npu":
        return _profile_fn_npu(
            fn, *args,
            label=label, warmup=warmup, iters=iters,
            trace_path=trace_path, row_limit=row_limit, sort_by=sort_by,
            **kwargs,
        )
    return _profile_fn_cuda(
        fn, *args,
        label=label, warmup=warmup, iters=iters,
        trace_path=trace_path, row_limit=row_limit, sort_by=sort_by,
        **kwargs,
    )


def _profile_fn_cuda(
    fn: Callable[..., Any],
    *args: Any,
    label: str | None,
    warmup: int,
    iters: int,
    trace_path: str | None,
    row_limit: int,
    sort_by: str,
    **kwargs: Any,
) -> torch.autograd.profiler_util.EventList:
    label = label or fn.__name__
    record_label = f"{fn.__name__}[{label}]"

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(iters):
            with record_function(record_label):
                fn(*args, **kwargs)
        torch.cuda.synchronize()

    averages = prof.key_averages()
    print(f"=== profile [{label}] (top {row_limit} by {sort_by}, iters={iters}) ===")
    print(averages.table(
        sort_by=sort_by,
        row_limit=row_limit,
        max_name_column_width=80,
    ))
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"chrome trace: {trace_path}")
    return averages


def _profile_fn_npu(
    fn: Callable[..., Any],
    *args: Any,
    label: str | None,
    warmup: int,
    iters: int,
    trace_path: str | None,
    row_limit: int,
    sort_by: str,
    **kwargs: Any,
) -> Any:
    import torch_npu  # noqa: F401  (registers torch.npu and the NPU profiler backend)
    from torch_npu.profiler import ProfilerActivity as NpuActivity
    from torch_npu.profiler import profile as npu_profile

    label = label or fn.__name__
    record_label = f"{fn.__name__}[{label}]"

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.npu.synchronize()

    on_trace_ready = None
    if trace_path:
        from torch_npu.profiler import tensorboard_trace_handler
        on_trace_ready = tensorboard_trace_handler(trace_path)

    with npu_profile(
        activities=[NpuActivity.CPU, NpuActivity.NPU],
        record_shapes=True,
        on_trace_ready=on_trace_ready,
    ) as prof:
        for _ in range(iters):
            with record_function(record_label):
                fn(*args, **kwargs)
        torch.npu.synchronize()

    npu_sort_by = sort_by.replace("cuda", "npu") if "cuda" in sort_by else sort_by

    averages = prof.key_averages()
    print(f"=== profile [{label}] (top {row_limit} by {npu_sort_by}, iters={iters}, backend=npu) ===")
    print(averages.table(
        sort_by=npu_sort_by,
        row_limit=row_limit,
        max_name_column_width=80,
    ))
    if trace_path:
        print(f"npu trace dir: {trace_path}")
    return averages
