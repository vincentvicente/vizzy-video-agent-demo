"""
Retry Router — conditional routing table mode (mode 3).

Every Fault emitted by QA is looked up in this table to decide which stage to
re-run. The routing table is **hard-coded** — the LLM never decides routing at
runtime. This is legibility-first made concrete: when a failure happens, a
glance at the table tells you exactly where the system goes next, fully
predictable.

The retry budget is also hard-coded: once exhausted → the whole run halts and
reports to the user (instead of looping forever).
"""
from __future__ import annotations

from typing import Literal, Optional
from schemas import QAReport, Fault, PipelineState


# Conditional routing table: fault_type → stage to retry from.
# Note that "spelling" covers two cases:
#   1. The model rendered misspelled text into the frame pixels (Hailuo/Seedance
#      ignored the negative prompt) → must go back to Director to fix the prompt.
#   2. The SRT subtitle is misspelled → regenerate via Editor.
# Default to Director (case 1 is more common, since the SRT comes from ElevenLabs
# VO transcription and won't misspell). An Editor re-run is cheap; a Director
# re-run is costly but actually fixes the problem. Choosing Director is the
# conservative call, backed by the retry budget as a safety net.
ROUTING_TABLE: dict[str, str] = {
    "spelling": "director",                     # rendered text in frame → Director rewrites prompt to strengthen anti-text
    "brand_consistency:color": "director",      # wrong palette → Director rewrites prompt with palette constraints
    "brand_consistency:tone": "strategist",     # wrong overall tone → Strategist re-derives brand understanding
    "claim_compliance": "strategist",           # non-compliant claim → Strategist fixes USP / rewrites VO
    "ranking_low": "halt",                      # poor overall but no specific cause → halt + report to user
    "ok": "none",                               # no retry needed
}


# Per-stage retry budget. Strict — once exceeded, halt; never loop forever.
DEFAULT_RETRY_BUDGET: dict[str, int] = {
    "strategist": 1,
    "director": 2,
    "clip_gen": 2,
    "editor": 1,
}


class RetryDecision:
    """Container for what the router decided."""
    def __init__(
        self,
        action: Literal["continue", "retry", "halt"],
        retry_from_stage: Optional[str] = None,
        reason: str = "",
        extra_constraints: Optional[dict] = None,
    ):
        self.action = action
        self.retry_from_stage = retry_from_stage
        self.reason = reason
        self.extra_constraints = extra_constraints or {}

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "retry_from_stage": self.retry_from_stage,
            "reason": self.reason,
            "extra_constraints": self.extra_constraints,
        }

    def __repr__(self):
        return f"RetryDecision({self.action}, from={self.retry_from_stage}, reason={self.reason!r})"


def decide(qa: QAReport, state: PipelineState) -> RetryDecision:
    """
    Look at the QA report and decide what happens next.

    Logic:
      - If overall_pass → continue (final video accepted)
      - Else: pick the FIRST block-severity fault, look up routing table, check budget
        - Budget OK → retry from mapped stage
        - Budget out → halt (legibility: tell user exactly which stage we gave up on)
    """
    if qa.overall_pass:
        return RetryDecision(action="continue", reason="QA passed")

    # Find first block-severity fault (or first fault if none are block)
    blocking = [f for f in qa.faults if f.severity == "block"]
    chosen = blocking[0] if blocking else (qa.faults[0] if qa.faults else None)

    if chosen is None:
        # ranking_low without specific fault
        if qa.ranking in ("D", "F"):
            chosen = Fault(
                fault_type="ranking_low",
                reason=f"Overall ranking {qa.ranking} with no specific fault flagged",
                severity="block",
            )
        else:
            return RetryDecision(action="continue", reason="QA warnings only, no block")

    target_stage = ROUTING_TABLE.get(chosen.fault_type, "halt")
    if target_stage in ("halt", "none"):
        return RetryDecision(
            action="halt",
            reason=f"No retry path for fault_type={chosen.fault_type} ({chosen.reason})",
        )

    # Budget check
    remaining = state.retry_budget.get(target_stage, 0)
    if remaining <= 0:
        return RetryDecision(
            action="halt",
            reason=(
                f"Retry budget exhausted for stage '{target_stage}' "
                f"(fault: {chosen.fault_type} — {chosen.reason})"
            ),
        )

    # Build extra constraints for the retry attempt — these get injected
    # into the target stage's prompt to nudge it away from the failure
    extra: dict = {}
    if chosen.fault_type == "brand_consistency:color":
        extra["palette_constraint"] = (
            "STRICT palette match required — the previous attempt drifted from brand colors. "
            "Re-emphasize palette anchors in every scene's prompt."
        )
    elif chosen.fault_type == "spelling":
        extra["anti_text_constraint"] = (
            "CRITICAL: previous attempt's video had rendered TEXT in the frame (model ignored negative prompt). "
            "Rewrite EVERY scene's prompt to be 100% wordless visuals. Add explicit phrases like "
            "'no text anywhere', 'no writing', 'no signage', 'no UI elements with words'. "
            "Pure pictorial composition only."
        )
    elif chosen.fault_type == "brand_consistency:tone":
        extra["tone_constraint"] = (
            "Brand tone mismatch — re-derive brand understanding with stricter tone tags."
        )
    elif chosen.fault_type == "claim_compliance":
        extra["compliance_constraint"] = (
            "Remove any visual/VO implying medical claims; rewrite USP to lifestyle framing."
        )

    return RetryDecision(
        action="retry",
        retry_from_stage=target_stage,
        reason=f"{chosen.fault_type}: {chosen.reason}",
        extra_constraints=extra,
    )


def consume_budget(state: PipelineState, stage: str) -> None:
    """Decrement retry budget after a retry is launched."""
    if stage in state.retry_budget:
        state.retry_budget[stage] = max(0, state.retry_budget[stage] - 1)
