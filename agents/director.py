"""
Director Agent.

Storyboard → DirectorOutput (per-scene Seedance 2.0 API call params).

Prototype 简化: 单一模型 (Seedance 2.0 Pro) for all scenes. 三层模型选择层退化,
LLM 只负责把 storyboard 的 visual_description 翻译成 cinematic video gen prompt.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from anthropic import Anthropic

from schemas import BrandUnderstanding, Storyboard, DirectorOutput, SceneAPICall
from utils.prompts import DIRECTOR_SYSTEM, director_user_prompt
from utils import video_model


_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def run_director(
    brand: BrandUnderstanding,
    storyboard: Storyboard,
    reference_uris: list[str],
    extra_constraints: Optional[dict] = None,
) -> DirectorOutput:
    """
    Convert each storyboard scene into a Seedance 2.0 API call spec.

    Reference URIs are attached to every scene call (Seedance Omni Reference will
    apply them as visual style/identity anchors).
    extra_constraints: retry_router 注入的强化指令 (e.g. anti-text/palette), 首次运行为 None.
    """
    client = Anthropic()
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=4000,
        system=DIRECTOR_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": director_user_prompt(
                    storyboard=storyboard.model_dump(),
                    brand=brand.model_dump(),
                    reference_uris=reference_uris,
                    extra_constraints=extra_constraints,
                ),
            }
        ],
    )
    raw = _strip_json_fences(resp.content[0].text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Director LLM returned non-JSON output:\n---\n{resp.content[0].text[:500]}\n---"
        ) from e

    # Cross-validate: api_calls count must match scene count
    if len(parsed["api_calls"]) != len(storyboard.scenes):
        raise ValueError(
            f"Director returned {len(parsed['api_calls'])} api_calls but storyboard has "
            f"{len(storyboard.scenes)} scenes"
        )

    # Force-inject reference URIs and durations from storyboard (don't trust LLM here —
    # these are deterministic data we already have)
    actual_model = video_model.model_label()  # 真实运行的模型, 不是写死的假名
    sb_by_id = {s.id: s for s in storyboard.scenes}
    for call in parsed["api_calls"]:
        sid = call["scene_id"]
        if sid not in sb_by_id:
            raise ValueError(f"Director invented scene_id '{sid}' not in storyboard")
        call["duration_s"] = sb_by_id[sid].duration_s
        call["reference_image_uris"] = reference_uris
        call["model"] = actual_model
        call["aspect_ratio"] = "9:16"
        call.setdefault(
            "negative_prompt",
            "text, logo, captions, subtitles, watermark, UI, words, letters",
        )

    # Recompute cost ourselves (don't trust LLM math) — 用实际模型的计费逻辑.
    estimated = video_model.estimate_cost([c["duration_s"] for c in parsed["api_calls"]])

    return DirectorOutput(api_calls=[SceneAPICall(**c) for c in parsed["api_calls"]],
                          estimated_cost_usd=estimated)


if __name__ == "__main__":
    # Smoke test
    from dotenv import load_dotenv
    from agents.strategist import run_strategist

    load_dotenv()
    out = run_strategist("https://goli.com/pages/goli-acv")
    brand = BrandUnderstanding(**out["brand"])
    sb = Storyboard(**out["storyboard"])
    d = run_director(brand, sb, reference_uris=[])
    print(json.dumps(d.model_dump(), indent=2, ensure_ascii=False))
