"""
Strict JSON schemas for every stage I/O.

设计原则: 每个 stage 都强制结构化输出，下游消费者可以可靠 parse。失败时整个 trace 都是
可读 JSON，方便下钻定位错误。
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

# Scene role enum (Schema B: LLM picks from this enum, decides sequence + duration)
SCENE_ROLES = Literal[
    "hook",            # 前 3 秒抓眼球
    "problem",         # 痛点
    "product_reveal",  # 产品出现
    "science",         # 为什么有效 / 机制
    "social_proof",    # 评论 / 评分 / 销量
    "comparison",      # vs 竞品
    "demo",            # 怎么用
    "cta",             # 行动召唤
]


class BrandUnderstanding(BaseModel):
    """Strategist 第一阶段输出 — 对品牌的认知。"""
    name: str = Field(description="brand name")
    usp: str = Field(description="unique selling proposition, 一句话")
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
    """单个 scene 的描述。"""
    id: str = Field(description="s1, s2, ...")
    role: SCENE_ROLES  # type: ignore
    duration_s: int = Field(ge=3, le=10, description="3-10 seconds per scene")
    visual_description: str = Field(description="what's on screen, detailed")
    voiceover: str = Field(description="VO line for this scene, written for ElevenLabs")
    purpose: str = Field(description="why this scene exists in the arc")


class Storyboard(BaseModel):
    """Strategist 第二阶段输出 — 完整分镜。"""
    total_duration_s: int = Field(ge=20, le=70, description="20-70s total")
    scenes: list[Scene] = Field(min_length=4, max_length=7)
    narrative_rationale: str = Field(
        description="why this role sequence works for this brand"
    )


class SceneAPICall(BaseModel):
    """Director 输出的单 scene API 调用参数."""
    scene_id: str
    # 实际运行的视频模型标签 (来自 utils.video_model.model_label), 不再写死 — trace 要记真实模型.
    model: str = "seedance-v1-pro"
    duration_s: int
    aspect_ratio: Literal["9:16"] = "9:16"
    prompt: str = Field(description="video gen prompt, no text/logo/captions")
    negative_prompt: str = "text, logo, captions, subtitles, watermark, UI, words, letters"
    reference_image_uris: list[str] = Field(
        default_factory=list,
        description="paths to user-uploaded reference images, max 9 per Seedance Omni"
    )
    seed: Optional[int] = None


class DirectorOutput(BaseModel):
    """Director 全部输出 — 每个 scene 一个 API call spec."""
    api_calls: list[SceneAPICall]
    estimated_cost_usd: float = Field(description="rough cost estimate before clip gen")


class Fault(BaseModel):
    """QA 发现的单个问题 — 喂给 retry_router。"""
    fault_type: Literal[
        "spelling",
        "brand_consistency:color",
        "brand_consistency:tone",
        "claim_compliance",
        "ranking_low",
        "ok",
    ]
    scene_id: Optional[str] = None
    reason: str
    severity: Literal["block", "warn", "info"] = "warn"


class QAReport(BaseModel):
    """QA 完整输出。"""
    spelling: dict = Field(description="{ok: bool, errors: [str]}")
    brand_consistency: dict = Field(description="{ok: bool, color_match: float, issues: [Fault]}")
    claim_compliance: dict = Field(description="{ok: bool, violations: [str]}")
    ranking: Literal["A", "B", "C", "D", "F"]
    faults: list[Fault] = Field(default_factory=list)
    overall_pass: bool


class PipelineState(BaseModel):
    """每个 stage 跑完都把当前完整 state 持久化, 供 trace + retry 用。"""
    run_id: str
    brand_url: str
    # text 输入模式: 用户直接粘品牌/产品文案, 不抓 URL. 二者择一作为 Strategist 的输入源.
    # 持久化它让 "retry from strategist" 和历史重载都能复用原始文本, 不会回退去 fetch 一个假 URL.
    brand_text: Optional[str] = None
    brand: Optional[BrandUnderstanding] = None
    storyboard: Optional[Storyboard] = None
    reference_image_uris: list[str] = Field(default_factory=list)
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
            "qa": 0,  # QA 不重试自己
        }
    )
    # stage → retry 约束 (retry_router 算出的 extra_constraints). 该 stage 重跑时读取并注入 prompt,
    # 跑完即清空. 这是 retry 闭环的载体: 让 "QA fault → 强化 prompt 重跑" 真正生效, 而不是用原 prompt 复跑.
    active_constraints: dict[str, dict] = Field(default_factory=dict)
    trace: list[dict] = Field(default_factory=list)
