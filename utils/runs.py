"""
Enumerate past pipeline runs from data/traces/ and rehydrate them into PipelineState.

Run persistence: each stage dumps its output as JSON to data/traces/<run_id>/<stage>.json
once it finishes. Here we reconstruct the PipelineState in reverse so the user can open any
old run from the sidebar to keep editing / regenerating it.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from schemas import (
    BrandUnderstanding,
    Storyboard,
    DirectorOutput,
    QAReport,
    PipelineState,
)

DATA_ROOT = Path(__file__).parent.parent / "data"
TRACE_ROOT = DATA_ROOT / "traces"
FINAL_ROOT = DATA_ROOT / "final"

# run_id looks like 20260521_163355_p2ul. Validate strictly before deleting to prevent
# path traversal from wiping out other directories.
_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9]{4}$")


def list_runs() -> list[dict]:
    """Return list of past run summaries, newest first."""
    if not TRACE_ROOT.exists():
        return []
    runs = []
    for d in sorted(TRACE_ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_id = d.name
        completed_stages = sorted([p.stem for p in d.glob("*.json") if p.is_file()])

        # Final video may not exist if pipeline halted before editor
        final_path = FINAL_ROOT / f"{run_id}.mp4"
        has_final = final_path.exists()

        # Try to extract brand name for display
        brand_name = None
        strategist_path = d / "strategist.json"
        if strategist_path.exists():
            try:
                with open(strategist_path) as f:
                    data = json.load(f)
                brand_name = data.get("brand", {}).get("name")
            except Exception:
                pass

        runs.append({
            "run_id": run_id,
            "brand_name": brand_name,
            "completed_stages": completed_stages,
            "has_final": has_final,
            "final_path": str(final_path) if has_final else None,
        })
    return runs


def delete_run(run_id: str) -> list[str]:
    """Permanently delete every file belonging to a run (irreversible).

    Cascade cleanup: the traces / clips / refs / voiceover directories + final/<run_id>.mp4.
    Returns the list of paths actually deleted (for UI display / logging).

    Raises ValueError if run_id is malformed — guards against accidentally deleting
    other things under data/.
    """
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"Refusing to delete: '{run_id}' is not a valid run_id")

    deleted: list[str] = []
    for sub in ("traces", "clips", "refs", "voiceover"):
        d = DATA_ROOT / sub / run_id
        # resolve + confirm it is still inside DATA_ROOT (belt and suspenders)
        if d.exists() and DATA_ROOT.resolve() in d.resolve().parents:
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(str(d))

    final_mp4 = FINAL_ROOT / f"{run_id}.mp4"
    if final_mp4.exists():
        final_mp4.unlink()
        deleted.append(str(final_mp4))

    return deleted


def load_state(run_id: str) -> Optional[PipelineState]:
    """Reconstruct a PipelineState from saved trace JSON files."""
    d = TRACE_ROOT / run_id
    if not d.exists():
        return None

    state = PipelineState(run_id=run_id, brand_url="(loaded from trace)")

    # Strategist (brand + storyboard)
    sp = d / "strategist.json"
    if sp.exists():
        try:
            with open(sp) as f:
                data = json.load(f)
            if "brand" in data:
                state.brand = BrandUnderstanding(**data["brand"])
            if "storyboard" in data:
                state.storyboard = Storyboard(**data["storyboard"])
            # Recover original URL if persisted
            page = data.get("page_snapshot", {})
            if isinstance(page, dict) and page.get("url"):
                state.brand_url = page["url"]
            # text mode: restore the original copy so a reloaded run can still retry from strategist correctly
            if data.get("brand_text"):
                state.brand_text = data["brand_text"]
        except Exception as e:
            print(f"[load_state] strategist parse failed: {e}")

    # Director (prompts + cost estimate)
    dp = d / "director.json"
    if dp.exists():
        try:
            with open(dp) as f:
                data = json.load(f)
            state.director_output = DirectorOutput(**data)
        except Exception as e:
            print(f"[load_state] director parse failed: {e}")

    # Clip Gen (paths to mp4s)
    cp = d / "clip_gen.json"
    if cp.exists():
        try:
            with open(cp) as f:
                data = json.load(f)
            state.clip_paths = data.get("clip_paths", {})
        except Exception as e:
            print(f"[load_state] clip_gen parse failed: {e}")

    # Editor (final video)
    ep = d / "editor.json"
    if ep.exists():
        try:
            with open(ep) as f:
                data = json.load(f)
            state.final_video_path = data.get("final_video_path")
        except Exception as e:
            print(f"[load_state] editor parse failed: {e}")

    # QA
    qp = d / "qa.json"
    if qp.exists():
        try:
            with open(qp) as f:
                data = json.load(f)
            state.qa_report = QAReport(**data)
        except Exception as e:
            print(f"[load_state] qa parse failed: {e}")

    # Reference image URIs (look for any uploaded refs dir)
    refs_dir = Path(__file__).parent.parent / "data" / "refs" / run_id
    if refs_dir.exists():
        state.reference_image_uris = sorted(str(p) for p in refs_dir.iterdir() if p.is_file())

    return state
