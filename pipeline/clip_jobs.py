"""
Background clip-generation jobs with live per-scene progress for the Streamlit UI.

Why this exists: run_clip_gen() blocks until every clip is done, so the UI could only
show an opaque spinner — the user couldn't tell which scene was generating or whether any
failed. Here we run generation on a daemon thread and publish per-scene status to a
module-level registry; the UI polls it each rerun and renders a live status list.

The registry lives at module scope (not in st.session_state) on purpose:
  - the worker thread has no Streamlit ScriptRunContext, so it must not touch session_state
  - module state survives Streamlit reruns AND view navigation, so progress keeps updating
    even if the user clicks away and comes back.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from schemas import DirectorOutput
from pipeline.clip_gen import run_clip_gen, ClipGenError

_LOCK = threading.Lock()
_JOBS: dict[str, "ClipJob"] = {}


@dataclass
class ClipJob:
    run_id: str
    total: int
    status: dict[str, str] = field(default_factory=dict)   # scene_id -> queued/submitting/running/done/failed
    results: dict[str, str] = field(default_factory=dict)  # scene_id -> mp4 path
    errors: dict[str, str] = field(default_factory=dict)   # scene_id -> error message
    done: bool = False


def _worker(job: ClipJob, director_output: DirectorOutput) -> None:
    def on_progress(scene_id: str, status: str) -> None:
        with _LOCK:
            job.status[scene_id] = status

    try:
        results = run_clip_gen(director_output, run_id=job.run_id, on_progress=on_progress)
        with _LOCK:
            job.results.update(results)
    except ClipGenError as e:
        # Partial failure: keep the successful clips, record per-scene errors.
        with _LOCK:
            job.results.update(e.results)
            for sid, err in e.errors:
                job.errors[sid] = str(err)
                job.status[sid] = "failed"
    except Exception as e:  # noqa: BLE001 — surface any unexpected failure to the UI
        with _LOCK:
            job.errors["_"] = str(e)
    finally:
        with _LOCK:
            for sid, p in job.results.items():
                job.status[sid] = "done"
            job.done = True


def start(run_id: str, director_output: DirectorOutput) -> bool:
    """Launch async generation for the given calls. No-op if a job is already running.

    Returns True if a new job was started, False if one was already active.
    """
    with _LOCK:
        existing = _JOBS.get(run_id)
        if existing and not existing.done:
            return False
        job = ClipJob(run_id=run_id, total=len(director_output.api_calls))
        for c in director_output.api_calls:
            job.status[c.scene_id] = "queued"
        _JOBS[run_id] = job
    threading.Thread(target=_worker, args=(job, director_output), daemon=True).start()
    return True


def snapshot(run_id: str) -> dict | None:
    """Consistent copy of a job's state for rendering, or None if no job exists."""
    with _LOCK:
        job = _JOBS.get(run_id)
        if not job:
            return None
        return {
            "status": dict(job.status),
            "results": dict(job.results),
            "errors": dict(job.errors),
            "done": job.done,
            "total": job.total,
        }


def clear(run_id: str) -> None:
    with _LOCK:
        _JOBS.pop(run_id, None)
