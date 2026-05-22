"""
Pipeline trace persistence.

每个 stage 跑完, 把完整 PipelineState 写到 data/traces/<run_id>/<stage>.json.
失败时可以根据 run_id 下钻看每一步发生了什么 — 这是 legibility-first 的物化。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

TRACE_ROOT = Path(__file__).parent.parent / "data" / "traces"


def save_trace(run_id: str, stage: str, payload: dict[str, Any]) -> Path:
    """Persist a stage's complete state."""
    run_dir = TRACE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{stage}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def append_trace_event(
    state: dict, stage: str, event: str, detail: dict | None = None
) -> None:
    """In-memory append to PipelineState.trace list."""
    state.setdefault("trace", []).append(
        {
            "ts": time.time(),
            "stage": stage,
            "event": event,
            "detail": detail or {},
        }
    )


def new_run_id() -> str:
    """Human-readable run id: YYYYMMDD_HHMMSS_<rand4>."""
    import random
    import string

    return time.strftime("%Y%m%d_%H%M%S_") + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=4)
    )
