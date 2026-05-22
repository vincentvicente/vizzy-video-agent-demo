"""
Enumerate past pipeline runs from data/traces/ and rehydrate them into PipelineState.

Run persistence: each stage dumps its output as JSON to data/traces/<run_id>/<stage>.json
once it finishes. Here we reconstruct the PipelineState in reverse so the user can open any
old run from the sidebar to keep editing / regenerating it.
"""
from __future__ import annotations

import json
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


def list_runs() -> list[dict]:
    """Return list of past run summaries, newest first."""
    if not TRACE_ROOT.exists():
        return []
    runs = []
    for d in sorted(TRACE_ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        run_id = d.name
        completed_stages = sorted(
            [p.stem for p in d.glob("*.json") if p.is_file() and p.stem != "meta"]
        )

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

        # User-set custom label (meta.json), if any — takes display priority over brand name
        label = None
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    label = (json.load(f).get("label") or "").strip() or None
            except Exception:
                pass

        runs.append({
            "run_id": run_id,
            "brand_name": brand_name,
            "label": label,
            "completed_stages": completed_stages,
            "has_final": has_final,
            "final_path": str(final_path) if has_final else None,
        })
    return runs


def set_run_label(run_id: str, label: str) -> None:
    """Set/clear a user-friendly display name for a run (stored in traces/<run_id>/meta.json).

    Does not touch the original stage traces. An empty label clears it.
    Raises ValueError on an unsafe run_id or a run that doesn't exist.
    """
    if (not run_id) or run_id in (".", "..") or any(c in run_id for c in ("/", "\\", "\0")):
        raise ValueError(f"Refusing: '{run_id}' is not a safe run id")
    run_dir = TRACE_ROOT / run_id
    if not run_dir.exists():
        raise ValueError(f"Run '{run_id}' not found")
    meta_path = run_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    label = (label or "").strip()
    if label:
        meta["label"] = label
    else:
        meta.pop("label", None)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def delete_run(run_id: str) -> list[str]:
    """Permanently delete every file belonging to a run (irreversible).

    Cascade cleanup: the traces / clips / refs / voiceover directories + final/<run_id>.mp4.
    Returns the list of paths actually deleted (for UI display / logging).

    Raises ValueError if run_id is unsafe — guards against path traversal wiping out
    other things under data/. Any run that shows up in list_runs() can be deleted; we
    only reject names that aren't a single safe path component.
    """
    if (not run_id) or run_id in (".", "..") or any(c in run_id for c in ("/", "\\", "\0")):
        raise ValueError(f"Refusing to delete: '{run_id}' is not a safe run id")

    deleted: list[str] = []
    data_root = DATA_ROOT.resolve()
    for sub in ("traces", "clips", "refs", "voiceover"):
        d = DATA_ROOT / sub / run_id
        # confirm the resolved target is a direct child inside DATA_ROOT (belt and suspenders)
        if d.exists() and data_root in d.resolve().parents:
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
            # restore user-edited storyboard generation prompt (if any)
            if data.get("storyboard_prompt"):
                state.storyboard_prompt = data["storyboard_prompt"]
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

    # Clip Gen — recover from trace if present, then reconcile with what's actually on disk.
    # Disk is ground truth: clips may exist even when clip_gen.json was never written
    # (e.g. generated via the per-scene / "generate missing clips" buttons), so a restart
    # must not "lose" them.
    cp = d / "clip_gen.json"
    if cp.exists():
        try:
            with open(cp) as f:
                state.clip_paths = json.load(f).get("clip_paths", {})
        except Exception as e:
            print(f"[load_state] clip_gen parse failed: {e}")
    clips_dir = DATA_ROOT / "clips" / run_id
    if clips_dir.exists():
        for mp4 in sorted(clips_dir.glob("*.mp4")):
            state.clip_paths.setdefault(mp4.stem, str(mp4))
    # Drop any recorded path whose file no longer exists
    state.clip_paths = {sid: p for sid, p in state.clip_paths.items() if Path(p).exists()}

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

    # Reference image URIs + their tags (look for any uploaded refs dir)
    refs_dir = DATA_ROOT / "refs" / run_id
    if refs_dir.exists():
        state.reference_image_uris = sorted(
            str(p) for p in refs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        tags_path = refs_dir / "tags.json"
        if tags_path.exists():
            try:
                with open(tags_path) as f:
                    by_name = json.load(f)
                state.reference_tags = {
                    uri: by_name[Path(uri).name]
                    for uri in state.reference_image_uris
                    if by_name.get(Path(uri).name)
                }
            except Exception as e:
                print(f"[load_state] reference tags parse failed: {e}")

    return state


def save_reference_tags(run_id: str, tags: dict[str, str]) -> None:
    """Persist reference-image tags to data/refs/<run_id>/tags.json, keyed by filename
    (stable across reloads). Empty tags are dropped."""
    refs_dir = DATA_ROOT / "refs" / run_id
    refs_dir.mkdir(parents=True, exist_ok=True)
    by_name = {Path(uri).name: tag.strip() for uri, tag in tags.items() if tag and tag.strip()}
    with open(refs_dir / "tags.json", "w", encoding="utf-8") as f:
        json.dump(by_name, f, ensure_ascii=False, indent=2)
