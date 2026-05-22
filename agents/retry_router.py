"""
Retry Router — 条件图模式 (你定的模式 3).

QA 输出的每个 Fault 都查这张表, 决定要回到哪个 stage 重做.
路由表是**写死的**, 不让 LLM 在线决定 — 这就是 legibility-first 的物化:
失败发生时, 你看一眼表就知道系统下一步会去哪里, 完全可预测.

Retry budget 也是写死的, 超过 → 整体 halt, 报给用户 (而不是无限循环).
"""
from __future__ import annotations

from typing import Literal, Optional
from schemas import QAReport, Fault, PipelineState


# 条件图: fault_type → stage to retry from
# 注意 "spelling" 有两种情况:
#   1. 模型在画面像素里生成了带错字的文本 (Hailuo/Seedance 无视 negative prompt) → 必须回 Director 改 prompt
#   2. SRT 字幕拼写错 → Editor 重生
# 默认走 Director (case 1 更常见, 因为 SRT 是 ElevenLabs VO 转的, 不会拼错). Editor 重跑代价小,
# Director 重跑代价大但能真正修问题. 选 Director 是有 retry budget 兜底的保守做法.
ROUTING_TABLE: dict[str, str] = {
    "spelling": "director",                     # 模型画面带错字 → Director 重写 prompt 强化 anti-text
    "brand_consistency:color": "director",      # palette 不对 → Director 重写 prompt 加 palette 约束
    "brand_consistency:tone": "strategist",     # 整体调性不对 → Strategist 重做 brand understanding
    "claim_compliance": "strategist",           # 违规 claim → Strategist 改 USP / 重写 VO
    "ranking_low": "halt",                      # 整体差但无具体原因 → halt + 报用户
    "ok": "none",                               # 不需要 retry
}


# 每个 stage 的 retry budget. Strict — 超了 halt, 永不无限循环.
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
