"""
Pipeline Orchestrator.

Runs the entire Vizzy pipeline end to end, persisting a trace at each stage and wiring
in the conditional-graph retry router.

This is not one long function — it's split into a separate entry point per stage, because
Streamlit is a stateful UI: each user action triggers one stage call, rather than running
straight through in a single pass.
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

    Pass brand_url → URL mode (scrape the web page); pass brand_text → text mode (the user
    pastes the copy directly).
    """
    return PipelineState(run_id=new_run_id(), brand_url=brand_url, brand_text=brand_text)


def stage_strategist(state: PipelineState, reference_count: int = 0) -> PipelineState:
    """Run Strategist (brand understanding + storyboard).

    One of two input sources: if state.brand_text is set, use text mode; otherwise scrape
    state.brand_url.
    """
    append_trace_event(state.model_dump(), "strategist", "start")
    # Consume the retry constraints (if this run is a retry from strategist). pop → cleared once used, so it doesn't pollute later runs.
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
        "brand_text": state.brand_text,  # Persist the original text for reuse on retry / reload
        "applied_constraints": constraints or {},
    })
    return state


def stage_director(state: PipelineState) -> PipelineState:
    """Run Director (storyboard → Seedance API calls)."""
    assert state.brand and state.storyboard, "Strategist must complete first"
    # Consume the retry constraints (if this run is a retry from director). pop → cleared once used.
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

    On partial failure, keep the successful clips (merged into state), then raise the
    ClipGenError to the caller — the user only needs to regenerate the failed scenes,
    not rerun the whole batch.
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
        # Attach the reinforced constraints to the target stage so they get injected into the prompt on its rerun — the key step that closes the retry loop.
        if decision.extra_constraints:
            state.active_constraints[decision.retry_from_stage] = decision.extra_constraints

    return state, decision
