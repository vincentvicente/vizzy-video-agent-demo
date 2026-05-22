"""
Strategist Agent.

Stage 1: URL → BrandUnderstanding
Stage 2: BrandUnderstanding → Storyboard

Each stage produces strict JSON output validated with Pydantic. On failure it
raises a ValidationError and the retry router takes over.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from anthropic import Anthropic

from schemas import BrandUnderstanding, Storyboard
from utils.url_fetcher import fetch_page_content
from utils.prompts import (
    STRATEGIST_BRAND_SYSTEM,
    STRATEGIST_STORYBOARD_SYSTEM,
    strategist_brand_user_prompt,
    strategist_storyboard_user_prompt,
)


_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _strip_json_fences(text: str) -> str:
    """Claude occasionally wraps JSON in ```json ... ``` fences — strip them."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _claude_json(system: str, user: str, max_tokens: int = 4096) -> dict:
    """Single LLM call, parse JSON, raise on bad output."""
    client = Anthropic()
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Strategist LLM returned non-JSON output:\n---\n{text[:500]}\n---"
        ) from e


def extract_brand_understanding(
    url: str, extra_constraints: Optional[dict] = None
) -> tuple[BrandUnderstanding, dict]:
    """
    Stage 1: fetch URL, extract structured brand understanding.

    Returns (validated BrandUnderstanding, raw page content dict for trace).
    extra_constraints: reinforcement instructions injected by retry_router
    (e.g. tone/compliance); None on the first run.
    """
    page = fetch_page_content(url)
    raw_json = _claude_json(
        system=STRATEGIST_BRAND_SYSTEM,
        user=strategist_brand_user_prompt(page, extra_constraints),
        max_tokens=1500,
    )
    brand = BrandUnderstanding(**raw_json)
    return brand, page


def extract_brand_from_text(
    raw_text: str, source_hint: str = "manual paste", extra_constraints: Optional[dict] = None
) -> tuple[BrandUnderstanding, dict]:
    """
    Stage 1 fallback: skip URL fetching, take raw text from user.

    Use case: the site has Akamai/Datadome-grade anti-bot protection we can't
    fetch through → the user copies the full text from their own browser and
    pastes it in. The Strategist workflow is unchanged (Claude reads the text and
    extracts brand info); only the input source shifts from requests.get to a
    user paste.
    """
    page = {
        "url": source_hint,
        "title": "",
        "meta_description": "",
        "headings": [],
        "body_text": raw_text[:5000],  # cap to avoid excessive token spend
        "image_urls": [],
    }
    raw_json = _claude_json(
        system=STRATEGIST_BRAND_SYSTEM,
        user=strategist_brand_user_prompt(page, extra_constraints),
        max_tokens=1500,
    )
    brand = BrandUnderstanding(**raw_json)
    return brand, page


def write_storyboard(
    brand: BrandUnderstanding, reference_count: int = 0,
    extra_constraints: Optional[dict] = None,
) -> Storyboard:
    """
    Stage 2: brand → storyboard (Schema B: LLM picks role sequence + duration).

    reference_count tells the model how many user-uploaded refs are available.
    extra_constraints: reinforcement instructions injected by retry_router
    (e.g. compliance VO rewrite); None on the first run.
    """
    raw_json = _claude_json(
        system=STRATEGIST_STORYBOARD_SYSTEM,
        user=strategist_storyboard_user_prompt(brand.model_dump(), reference_count, extra_constraints),
        max_tokens=3000,
    )
    sb = Storyboard(**raw_json)

    # Hard constraint validation beyond Pydantic
    if sb.scenes[0].role != "hook":
        raise ValueError(
            f"Storyboard violates constraint: first scene must be 'hook', got '{sb.scenes[0].role}'"
        )
    if sb.scenes[-1].role != "cta":
        raise ValueError(
            f"Storyboard violates constraint: last scene must be 'cta', got '{sb.scenes[-1].role}'"
        )
    # Per-scene durations are the source of truth — overwrite LLM's declared total
    # (LLMs are unreliable at arithmetic but reliable at structure).
    actual_total = sum(s.duration_s for s in sb.scenes)
    sb.total_duration_s = actual_total
    if not (25 <= actual_total <= 45):
        raise ValueError(
            f"Storyboard total duration {actual_total}s outside [25, 45] bound"
        )

    return sb


def run_strategist(url: str, reference_count: int = 0) -> dict:
    """
    Convenience wrapper: returns dict ready to merge into PipelineState.
    """
    brand, page = extract_brand_understanding(url)
    storyboard = write_storyboard(brand, reference_count=reference_count)
    return {
        "brand": brand.model_dump(),
        "storyboard": storyboard.model_dump(),
        "_page_snapshot": page,  # for trace
    }


if __name__ == "__main__":
    # Quick smoke test (requires ANTHROPIC_API_KEY)
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    url = sys.argv[1] if len(sys.argv) > 1 else "https://goli.com/pages/goli-acv"
    result = run_strategist(url, reference_count=0)
    print(json.dumps(result, indent=2, ensure_ascii=False))
