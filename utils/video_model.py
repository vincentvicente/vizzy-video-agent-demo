"""
Single source of truth for the video model + provider.

Three providers are supported, selected by VIDEO_PROVIDER:
  - "fal"     → fal.ai endpoints (Seedance v1 / Hailuo), submit-and-block API.
  - "volcano" → Volcengine Ark (Seedance 2.0), async task + polling API.
  - "atlas"   → Atlas Cloud (Seedance 2.0), POST + poll API.

This module consolidates "which provider / which model id / human-readable label /
billing logic / cost estimation" in one place; clip_gen and director both pull from
here, so the trace always records the model that actually ran (no hardcoded fakes).
"""
from __future__ import annotations

import os

# ---------- Provider selection ----------
# "fal", "volcano", or "atlas". Overridden at runtime via set_provider() (UI picker).
VIDEO_PROVIDER = os.environ.get("VIDEO_PROVIDER", "volcano").lower()

# ---------- fal config ----------
# fal video model endpoint — determines which model runs under the fal provider.
FAL_VIDEO_MODEL = os.environ.get(
    "FAL_VIDEO_MODEL", "fal-ai/bytedance/seedance/v1/pro/image-to-video"
)

# ---------- Volcengine Ark (Seedance 2.0) config ----------
# China region defaults; override ARK_BASE_URL + VOLCANO_VIDEO_MODEL for BytePlus international:
#   ARK_BASE_URL=https://ark.ap-southeast.bytepluses.com/api/v3
#   VOLCANO_VIDEO_MODEL=dreamina-seedance-2-0-260128
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_API_KEY = os.environ.get("ARK_API_KEY")
VOLCANO_VIDEO_MODEL = os.environ.get("VOLCANO_VIDEO_MODEL", "doubao-seedance-2-0-260128")
# Ark output settings (Seedance 2.0)
VOLCANO_RESOLUTION = os.environ.get("VOLCANO_RESOLUTION", "1080p")  # 480p/720p/1080p/2K
# Role attached to reference images. "reference_image" = identity/style anchor (best for
# cross-scene consistency). Set empty to omit (then the image is a literal first frame).
VOLCANO_IMAGE_ROLE = os.environ.get("VOLCANO_IMAGE_ROLE", "reference_image")

# ---------- Atlas Cloud (Seedance 2.0) config ----------
ATLAS_API_KEY = os.environ.get("ATLAS_API_KEY")
ATLAS_BASE_URL = os.environ.get("ATLAS_BASE_URL", "https://api.atlascloud.ai/api/v1/model")
ATLAS_VIDEO_MODEL = os.environ.get(
    "ATLAS_VIDEO_MODEL", "bytedance/seedance-2.0/image-to-video"
)

# ---------- All available provider+model combos for the UI picker ----------
VIDEO_OPTIONS: list[dict[str, str]] = [
    {"provider": "volcano", "model": "doubao-seedance-2-0-260128",                  "label": "Seedance 2.0 (Volcano)"},
    {"provider": "atlas",   "model": "bytedance/seedance-2.0/image-to-video",       "label": "Seedance 2.0 (Atlas)"},
    {"provider": "atlas",   "model": "bytedance/seedance-2.0/text-to-video",        "label": "Seedance 2.0 text-only (Atlas)"},
    {"provider": "atlas",   "model": "bytedance/seedance-2.0-fast/image-to-video",  "label": "Seedance 2.0 Fast (Atlas)"},
    {"provider": "fal",     "model": "fal-ai/bytedance/seedance/v1/pro/image-to-video", "label": "Seedance v1 Pro (fal)"},
    {"provider": "fal",     "model": "fal-ai/minimax/hailuo-02-fast/image-to-video","label": "Hailuo 02 Fast (fal)"},
]


def set_provider(provider: str, model_id: str | None = None) -> None:
    """Switch provider (and optionally model) at runtime — called by the UI picker."""
    global VIDEO_PROVIDER, FAL_VIDEO_MODEL, VOLCANO_VIDEO_MODEL, ATLAS_VIDEO_MODEL
    VIDEO_PROVIDER = provider.lower()
    if model_id:
        if VIDEO_PROVIDER == "fal":
            FAL_VIDEO_MODEL = model_id
        elif VIDEO_PROVIDER == "volcano":
            VOLCANO_VIDEO_MODEL = model_id
        elif VIDEO_PROVIDER == "atlas":
            ATLAS_VIDEO_MODEL = model_id


def active_model_id() -> str:
    """The model id of the currently selected provider."""
    if VIDEO_PROVIDER == "volcano":
        return VOLCANO_VIDEO_MODEL
    if VIDEO_PROVIDER == "atlas":
        return ATLAS_VIDEO_MODEL
    return FAL_VIDEO_MODEL


def model_label(model_id: str | None = None) -> str:
    """Translate a provider model id into a human-readable label, e.g. 'hailuo-02-fast',
    'seedance-v1-pro', 'seedance-2.0', 'seedance-2.0-fast'.

    This is the real model name written into the trace and shown to the user.
    """
    ml = (model_id or active_model_id()).lower()
    if "hailuo" in ml:
        tier = "fast" if "fast" in ml else "standard"
        return f"hailuo-02-{tier}"
    if "seedance" in ml:
        # Volcano ids look like doubao-seedance-2-0-260128 / ...-fast-...
        if "seedance-2" in ml or "2.0" in ml or "/v2" in ml:
            return "seedance-2.0-fast" if "fast" in ml else "seedance-2.0"
        ver = "v1"
        tier = "fast" if "fast" in ml else ("pro" if "pro" in ml else "standard")
        return f"seedance-{ver}-{tier}"
    return model_id or active_model_id()


def billed_seconds(duration_s: float, model_id: str | None = None) -> int:
    """Billed duration for a single clip.

    Hailuo 02: 6s minimum, otherwise 10s. Seedance v1 (fal): 5s or 10s tier.
    Seedance 2.0 (volcano): continuous 4-15s, billed by the requested (clamped) duration.
    """
    ml = (model_id or active_model_id()).lower()
    if "hailuo" in ml:
        return 6 if duration_s <= 6 else 10
    if "seedance-2" in ml or "2.0" in ml:
        return max(4, min(15, int(round(duration_s))))
    if "seedance" in ml:
        return 5 if duration_s <= 5 else 10
    # Unknown model: bill by actual seconds (conservative)
    return max(1, int(round(duration_s)))


def cost_per_second(model_id: str | None = None) -> float:
    """Cost per second (USD). An explicit VIDEO_COST_PER_SECOND (or legacy FAL_COST_PER_SECOND)
    override takes priority; otherwise the model's default price.

    Defaults are rough pre-checkpoint estimates and vary with provider pricing:
      Hailuo 02 Standard 768p ≈ $0.045/s
      Seedance v1 Fast ≈ $0.24/s, Seedance v1 Pro ≈ $0.30/s
      Seedance 2.0 ≈ $0.30/s (override VIDEO_COST_PER_SECOND with your real Ark price)
    """
    env = os.environ.get("VIDEO_COST_PER_SECOND") or os.environ.get("FAL_COST_PER_SECOND")
    if env:
        return float(env)
    ml = (model_id or active_model_id()).lower()
    if "hailuo" in ml:
        return 0.045
    if "seedance-2" in ml or "2.0" in ml:
        return 0.30
    if "seedance" in ml:
        return 0.24 if "fast" in ml else 0.30
    return 0.05


def estimate_cost(durations_s: list[float], model_id: str | None = None) -> float:
    """Rough total cost estimate before clip gen = Σ (each clip's billed duration × cost per second)."""
    total_billed = sum(billed_seconds(d, model_id) for d in durations_s)
    return round(total_billed * cost_per_second(model_id), 2)
