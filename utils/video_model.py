"""
视频模型的单一事实源 (single source of truth).

之前的问题: schema 把 model 硬写成 "seedance-2.0-pro", director 也强行覆盖成这个名字,
但实际跑的是 FAL_VIDEO_MODEL 指定的模型 (默认 Seedance v1 pro, .env 里常覆盖成 Hailuo).
成本估算用的是 Hailuo 计费逻辑却标 Seedance —— trace 记录的模型名和计费全是假的.

这个模块把"实际模型 id / 人类可读标签 / 计费逻辑 / 成本估算"集中到一处, clip_gen 和 director
都从这里取, 保证 trace 里记录的就是真实运行的模型.
"""
from __future__ import annotations

import os

# fal 视频模型 endpoint — 实际运行的模型由它决定. clip_gen submit 用的就是这个.
FAL_VIDEO_MODEL = os.environ.get(
    "FAL_VIDEO_MODEL", "fal-ai/bytedance/seedance/v1/pro/image-to-video"
)


def model_label(model_id: str | None = None) -> str:
    """把 fal endpoint 翻译成人类可读标签, 如 'hailuo-02-fast' / 'seedance-v1-pro'.

    这是写进 trace + 展示给用户的真实模型名, 不再是写死的假名字.
    """
    ml = (model_id or FAL_VIDEO_MODEL).lower()
    if "hailuo" in ml:
        tier = "fast" if "fast" in ml else "standard"
        return f"hailuo-02-{tier}"
    if "seedance" in ml:
        ver = "v2" if ("/v2" in ml or "2.0" in ml) else "v1"
        tier = "fast" if "fast" in ml else ("pro" if "pro" in ml else "standard")
        return f"seedance-{ver}-{tier}"
    return model_id or FAL_VIDEO_MODEL


def billed_seconds(duration_s: float, model_id: str | None = None) -> int:
    """单 clip 的计费时长 (各模型按固定档计费, 向上取整到下一档).

    Hailuo 02: 6s 起步, 否则 10s. Seedance: 5s 或 10s.
    """
    ml = (model_id or FAL_VIDEO_MODEL).lower()
    if "hailuo" in ml:
        return 6 if duration_s <= 6 else 10
    if "seedance" in ml:
        return 5 if duration_s <= 5 else 10
    # 未知模型: 按真实秒数计 (保守)
    return max(1, int(round(duration_s)))


def cost_per_second(model_id: str | None = None) -> float:
    """每秒成本 (USD). FAL_COST_PER_SECOND 显式覆盖优先, 否则按模型默认价.

    默认价 (随 fal 定价变动, 仅做 γ checkpoint 前的粗估):
      Hailuo 02 Standard 768p ≈ $0.045/s
      Seedance Fast ≈ $0.24/s, Seedance Pro ≈ $0.30/s
    """
    env = os.environ.get("FAL_COST_PER_SECOND")
    if env:
        return float(env)
    ml = (model_id or FAL_VIDEO_MODEL).lower()
    if "hailuo" in ml:
        return 0.045
    if "seedance" in ml:
        return 0.24 if "fast" in ml else 0.30
    return 0.05


def estimate_cost(durations_s: list[float], model_id: str | None = None) -> float:
    """clip gen 前的总成本粗估 = Σ 每个 clip 计费时长 × 每秒成本."""
    total_billed = sum(billed_seconds(d, model_id) for d in durations_s)
    return round(total_billed * cost_per_second(model_id), 2)
