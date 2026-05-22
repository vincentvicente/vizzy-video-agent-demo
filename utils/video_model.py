"""
Single source of truth for the video model.

The previous problem: the schema hardcoded model to "seedance-2.0-pro" and the director
also forced it to this name, but what actually ran was the model specified by
FAL_VIDEO_MODEL (default Seedance v1 pro, often overridden to Hailuo in .env).
Cost estimation used Hailuo's billing logic while labeling it Seedance — so the model
name and billing recorded in the trace were both fake.

This module consolidates "actual model id / human-readable label / billing logic /
cost estimation" in one place; clip_gen and director both pull from here, guaranteeing
the trace records the model that actually ran.
"""
from __future__ import annotations

import os

# fal video model endpoint — it determines the model that actually runs. clip_gen submits with this.
FAL_VIDEO_MODEL = os.environ.get(
    "FAL_VIDEO_MODEL", "fal-ai/bytedance/seedance/v1/pro/image-to-video"
)


def model_label(model_id: str | None = None) -> str:
    """Translate a fal endpoint into a human-readable label, e.g. 'hailuo-02-fast' / 'seedance-v1-pro'.

    This is the real model name written into the trace and shown to the user, no longer a hardcoded fake one.
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
    """Billed duration for a single clip (each model bills in fixed tiers, rounded up to the next tier).

    Hailuo 02: 6s minimum, otherwise 10s. Seedance: 5s or 10s.
    """
    ml = (model_id or FAL_VIDEO_MODEL).lower()
    if "hailuo" in ml:
        return 6 if duration_s <= 6 else 10
    if "seedance" in ml:
        return 5 if duration_s <= 5 else 10
    # Unknown model: bill by actual seconds (conservative)
    return max(1, int(round(duration_s)))


def cost_per_second(model_id: str | None = None) -> float:
    """Cost per second (USD). An explicit FAL_COST_PER_SECOND override takes priority, otherwise the model's default price.

    Default prices (vary with fal's pricing, only a rough estimate before the γ checkpoint):
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
    """Rough total cost estimate before clip gen = Σ (each clip's billed duration × cost per second)."""
    total_billed = sum(billed_seconds(d, model_id) for d in durations_s)
    return round(total_billed * cost_per_second(model_id), 2)
