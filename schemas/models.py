"""
Strict JSON schemas for every stage I/O.

Design principle: every stage is forced to emit structured output so downstream
consumers can parse it reliably. On failure, the entire trace is readable JSON,
making it easy to drill down and locate the error.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

# Scene role enum (Schema B: LLM picks from this enum, decides sequence + duration)
SCENE_ROLES = Literal[
    "hook",            # grab attention in the first 3 seconds
    "problem",         # pain point
    "product_reveal",  # product appears
    "science",         # why it works / mechanism
    "social_proof",    # reviews / ratings / sales
    "comparison",      # vs competitors
    "demo",            # how to use it
    "cta",             # call to action
]


class BrandUnderstanding(BaseModel):
    """Strategist stage-one output — its understanding of the brand."""
    name: str = Field(description="brand name")
    usp: str = Field(description="unique selling proposition, one sentence")
    target_audience: str = Field(description="who this is for")
    tone: list[str] = Field(description="brand tone tags, e.g. ['clinical','minimal']")
    palette: list[str] = Field(description="hex color palette, 2-5 colors")
    product_visual_keywords: list[str] = Field(
        description="visual identifiers of the hero product, e.g. ['red gummy', 'white bottle']"
    )
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description="claims that must NOT appear (regulatory / compliance), inferred from category"
    )


class Scene(BaseModel):
    """Description of a single scene."""
    id: str = Field(description="s1, s2, ...")
    role: SCENE_ROLES  # type: ignore
    duration_s: int = Field(ge=3, le=10, description="3-10 seconds per scene")
    visual_description: str = Field(description="what's on screen, detailed")
    voiceover: str = Field(description="VO line for this scene, written for ElevenLabs")
    purpose: str = Field(description="why this scene exists in the arc")


class Storyboard(BaseModel):
    """Strategist stage-two output — the complete storyboard."""
    total_duration_s: int = Field(ge=20, le=70, description="20-70s total")
    scenes: list[Scene] = Field(min_length=4, max_length=7)
    narrative_rationale: str = Field(
        description="why this role sequence works for this brand"
    )


class SceneAPICall(BaseModel):
    """API call parameters for a single scene, produced by the Director."""
    scene_id: str
    # Label of the video model actually running (from utils.video_model.model_label),
    # no longer hardcoded — the trace must record the real model.
    model: str = "seedance-v1-pro"
    duration_s: int
    aspect_ratio: Literal["9:16"] = "9:16"
    prompt: str = Field(description="video gen prompt, no text/logo/captions")
    negative_prompt: str = "text, logo, brand mark, competitor logo, emblem, signage, captions, subtitles, watermark, UI, words, letters"
    reference_image_uris: list[str] = Field(
        default_factory=list,
        description="paths to user-uploaded reference images, max 9 per Seedance Omni"
    )
    seed: Optional[int] = None


class DirectorOutput(BaseModel):
    """Full Director output — one API call spec per scene."""
    api_calls: list[SceneAPICall]
    estimated_cost_usd: float = Field(description="rough cost estimate before clip gen")


class Fault(BaseModel):
    """A single issue found by QA — fed to retry_router."""
    fault_type: Literal[
        "spelling",
        "unwanted_content",   # model rendered an unwanted logo / watermark / competitor mark / signage
        "brand_consistency:color",
        "brand_consistency:tone",
        "claim_compliance",
        "ranking_low",
        "other",   # catch-all for QA findings outside the fixed vocabulary (routes to halt)
        "ok",
    ]
    scene_id: Optional[str] = None
    reason: str
    severity: Literal["block", "warn", "info"] = "warn"


class QAReport(BaseModel):
    """Full QA output."""
    spelling: dict = Field(description="{ok: bool, errors: [str]}")
    brand_consistency: dict = Field(description="{ok: bool, color_match: float, issues: [Fault]}")
    claim_compliance: dict = Field(description="{ok: bool, violations: [str]}")
    ranking: Literal["A", "B", "C", "D", "F"]
    faults: list[Fault] = Field(default_factory=list)
    overall_pass: bool


class PipelineState(BaseModel):
    """After each stage finishes, persist the full current state for trace + retry."""
    run_id: str
    brand_url: str
    # Text input mode: the user pastes brand/product copy directly instead of fetching a URL.
    # One of the two serves as the Strategist's input source.
    # Persisting it lets "retry from strategist" and history reloads reuse the original text,
    # rather than falling back to fetching a bogus URL.
    brand_text: Optional[str] = None
    brand: Optional[BrandUnderstanding] = None
    storyboard: Optional[Storyboard] = None
    reference_image_uris: list[str] = Field(default_factory=list)
    # uri → user tag describing what the image is (e.g. "hero product", "mascot").
    # Fed to storyboard generation so the Strategist can plan scenes around named assets.
    reference_tags: dict[str, str] = Field(default_factory=dict)
    # User-edited storyboard generation system prompt (None = use the built-in default).
    storyboard_prompt: Optional[str] = None
    director_output: Optional[DirectorOutput] = None
    clip_paths: dict[str, str] = Field(default_factory=dict)  # scene_id → mp4 path
    final_video_path: Optional[str] = None
    qa_report: Optional[QAReport] = None
    retry_budget: dict[str, int] = Field(
        default_factory=lambda: {
            "strategist": 1,
            "director": 2,
            "clip_gen": 2,
            "editor": 1,
            "qa": 0,  # QA does not retry itself
        }
    )
    # stage → retry constraints (extra_constraints computed by retry_router). When that stage
    # reruns, they are read and injected into the prompt, then cleared once the run completes.
    # This is the carrier of the retry loop: it makes "QA fault → rerun with a strengthened prompt"
    # actually take effect, rather than rerunning with the original prompt.
    active_constraints: dict[str, dict] = Field(default_factory=dict)
    trace: list[dict] = Field(default_factory=list)
