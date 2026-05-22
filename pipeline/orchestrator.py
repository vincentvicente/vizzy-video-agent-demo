"""
Pipeline Orchestrator.

跑通整个 Vizzy pipeline, 在每个 stage 持久化 trace, 接入条件图 retry router.

不是一个长 function — 拆成各 stage 的独立 entry point, 因为 Streamlit 是 stateful UI,
每个用户操作触发一次 stage 调用, 而不是一口气跑到底.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from schemas import (
    BrandUnderstanding,
    Storyboard,
    DirectorOutput,
    QAReport,
    PipelineState,
)
from agents.strategist import (
    extract_brand_understanding,
    extract_brand_from_text,
    write_storyboard,
)
from agents.director import run_director
from agents.qa import run_qa
from agents import retry_router
from pipeline.clip_gen import run_clip_gen, ClipGenError
from pipeline.editor import run_editor
from utils.trace import save_trace, append_trace_event, new_run_id


def new_pipeline(brand_url: str = "", brand_text: Optional[str] = None) -> PipelineState:
    """Initialize a fresh run.

    传 brand_url → URL 模式 (抓网页); 传 brand_text → text 模式 (用户直接粘文案).
    """
    return PipelineState(run_id=new_run_id(), brand_url=brand_url, brand_text=brand_text)


def stage_strategist(state: PipelineState, reference_count: int = 0) -> PipelineState:
    """Run Strategist (brand understanding + storyboard).

    输入源二选一: state.brand_text 有值走 text 模式, 否则抓 state.brand_url.
    """
    append_trace_event(state.model_dump(), "strategist", "start")
    # 消费 retry 约束 (若本次是 retry from strategist). pop → 跑完即清, 不污染后续运行.
    constraints = state.active_constraints.pop("strategist", None)
    if state.brand_text:
        brand, page = extract_brand_from_text(
            state.brand_text,
            source_hint=state.brand_url or "text input",
            extra_constraints=constraints,
        )
    else:
        brand, page = extract_brand_understanding(state.brand_url, extra_constraints=constraints)
    storyboard = write_storyboard(brand, reference_count=reference_count, extra_constraints=constraints)
    state.brand = brand
    state.storyboard = storyboard
    save_trace(state.run_id, "strategist", {
        "brand": brand.model_dump(),
        "storyboard": storyboard.model_dump(),
        "page_snapshot": page,
        "brand_text": state.brand_text,  # 持久化原始文本, 供 retry / 重载复用
        "applied_constraints": constraints or {},
    })
    return state


def stage_director(state: PipelineState) -> PipelineState:
    """Run Director (storyboard → Seedance API calls)."""
    assert state.brand and state.storyboard, "Strategist must complete first"
    # 消费 retry 约束 (若本次是 retry from director). pop → 跑完即清.
    constraints = state.active_constraints.pop("director", None)
    director_output = run_director(
        brand=state.brand,
        storyboard=state.storyboard,
        reference_uris=state.reference_image_uris,
        extra_constraints=constraints,
    )
    state.director_output = director_output
    trace_payload = director_output.model_dump()
    trace_payload["applied_constraints"] = constraints or {}
    save_trace(state.run_id, "director", trace_payload)
    return state


def stage_clip_gen(
    state: PipelineState,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> PipelineState:
    """Run Clip Gen (parallel fal.ai calls).

    部分失败时保留已成功的 clip (merge 进 state), 再把 ClipGenError 抛给 caller —
    用户只需重生失败的 scene, 不必整批重跑.
    """
    assert state.director_output, "Director must complete first"
    try:
        clip_paths = run_clip_gen(
            state.director_output,
            run_id=state.run_id,
            on_progress=on_progress,
        )
    except ClipGenError as e:
        state.clip_paths.update(e.results)
        save_trace(state.run_id, "clip_gen", {
            "clip_paths": state.clip_paths,
            "errors": {sid: str(err) for sid, err in e.errors},
        })
        raise
    state.clip_paths.update(clip_paths)
    save_trace(state.run_id, "clip_gen", {"clip_paths": state.clip_paths})
    return state


def stage_editor(state: PipelineState) -> PipelineState:
    """Run Editor (ffmpeg + ElevenLabs + subtitles)."""
    assert state.clip_paths and state.storyboard, "Clip gen and storyboard must complete first"
    final_path = run_editor(
        clip_paths=state.clip_paths,
        storyboard=state.storyboard,
        run_id=state.run_id,
        burn_subtitles=True,
    )
    state.final_video_path = str(final_path)
    save_trace(state.run_id, "editor", {"final_video_path": str(final_path)})
    return state


def stage_qa(state: PipelineState) -> tuple[PipelineState, retry_router.RetryDecision]:
    """Run QA + consult retry router for next action."""
    assert state.final_video_path and state.brand and state.storyboard
    qa_report = run_qa(
        final_video_path=Path(state.final_video_path),
        brand=state.brand,
        storyboard=state.storyboard,
    )
    state.qa_report = qa_report
    save_trace(state.run_id, "qa", qa_report.model_dump())

    decision = retry_router.decide(qa_report, state)
    save_trace(state.run_id, "retry_decision", decision.to_dict())

    if decision.action == "retry":
        retry_router.consume_budget(state, decision.retry_from_stage)
        # 把强化约束挂到目标 stage 上, 等它重跑时注入 prompt — 这是 retry 闭环的关键一步.
        if decision.extra_constraints:
            state.active_constraints[decision.retry_from_stage] = decision.extra_constraints

    return state, decision
