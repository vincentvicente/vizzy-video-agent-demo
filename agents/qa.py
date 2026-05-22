"""
QA Agent — Claude vision 抽帧 + 检查.

抽 N 帧从 final video, 喂给 Claude vision + brand/storyboard 上下文, 拿到 structured QAReport.

检查维度 (per PRD output #5):
  - spelling (frame OCR via vision)
  - brand_consistency (color + tone)
  - claim_compliance
  - ranking
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from anthropic import Anthropic

from schemas import BrandUnderstanding, Storyboard, QAReport, Fault
from utils.prompts import QA_SYSTEM, qa_user_prompt


_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_frames(video_path: Path, n_frames: int = 4) -> list[Path]:
    """Sample n evenly-spaced frames from the video."""
    # Get duration
    p = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(p.stdout.strip())

    tmp = Path(tempfile.mkdtemp(prefix="qa_frames_"))
    frame_paths: list[Path] = []
    # Sample at uniform intervals (avoid first/last 5%)
    for i in range(n_frames):
        t = duration * (0.05 + 0.9 * (i / max(1, n_frames - 1)))
        out = tmp / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
                "-vframes", "1", "-q:v", "3",
                str(out),
            ],
            capture_output=True, check=True,
        )
        frame_paths.append(out)
    return frame_paths


def _frame_to_block(path: Path) -> dict:
    """Build an Anthropic vision content block."""
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64,
        },
    }


def run_qa(
    final_video_path: Path,
    brand: BrandUnderstanding,
    storyboard: Storyboard,
    n_frames: int = 4,
) -> QAReport:
    """
    Run QA on the final video.

    Returns QAReport (validated). Caller routes faults via retry_router.
    """
    frames = _extract_frames(final_video_path, n_frames=n_frames)

    user_text = qa_user_prompt(brand=brand.model_dump(), storyboard=storyboard.model_dump())
    content_blocks: list[dict] = [_frame_to_block(p) for p in frames]
    content_blocks.append({"type": "text", "text": user_text})

    client = Anthropic()
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=2000,
        system=QA_SYSTEM,
        messages=[{"role": "user", "content": content_blocks}],
    )
    raw = _strip_json_fences(resp.content[0].text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"QA LLM returned non-JSON:\n---\n{resp.content[0].text[:500]}\n---"
        ) from e

    # Normalize faults list — flatten brand_consistency.issues into top-level faults
    faults_list = parsed.get("faults", []) or []
    for issue in parsed.get("brand_consistency", {}).get("issues", []):
        faults_list.append(issue)
    parsed["faults"] = faults_list

    # Compute overall_pass server-side (don't trust LLM judgment alone)
    has_block = any(f.get("severity") == "block" for f in faults_list)
    parsed["overall_pass"] = not has_block

    return QAReport(**parsed)


if __name__ == "__main__":
    print("QA module loaded. Use run_qa(final_video_path, brand, storyboard).")
