"""
Clip Generation runner.

Calls fal.ai Seedance 2.0 Pro image-to-video in parallel, turning each scene's
SceneAPICall into an actual mp4 file.

Design notes:
- Uses ThreadPoolExecutor for parallelism (Streamlit is a sync environment, so we avoid asyncio)
- A single clip failure does NOT fail the whole stage outright — the successful clips are
  preserved and a ClipGenError is raised for the retry router to handle
- Each clip is downloaded to data/clips/<run_id>/<scene_id>.mp4
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fal_client
import requests

from schemas import DirectorOutput, SceneAPICall
from utils import video_model


# Single source of truth: the fal model actually used at runtime (see utils/video_model.py)
_FAL_MODEL = video_model.FAL_VIDEO_MODEL
_CLIPS_ROOT = Path(__file__).parent.parent / "data" / "clips"


class ClipGenError(RuntimeError):
    """Raised when some clips fail to generate. The key idea: carry the already-successful
    clips out with the error so the caller can keep them.

    The user only needs to regenerate the failed scenes, instead of rerunning the whole
    batch (~$1 each). This is the core of saving cost.
    """
    def __init__(self, results: dict[str, str], errors: list[tuple[str, Exception]]):
        self.results = results
        self.errors = errors
        detail = "\n".join(f"  {sid}: {e}" for sid, e in errors)
        super().__init__(
            f"Clip gen failed for {len(errors)} scene(s) "
            f"({len(results)} succeeded, kept):\n{detail}"
        )


def _to_data_uri(path: str) -> str:
    """Encode a local file as a base64 data URI.

    We use a data URI instead of fal_client.upload_file() — upload_file goes through fal
    storage and needs the storage_write permission, which many fal keys (especially free
    tier) don't have. A data URI is embedded directly in the request body, bypasses fal
    storage, and works with any key.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_args(call: SceneAPICall) -> dict:
    """Build the input dict — schema differs by model family."""
    # Take the first reference image and encode it as a data URI to avoid fal storage permission issues
    image_url = None
    if call.reference_image_uris:
        first = call.reference_image_uris[0]
        if first.startswith(("http://", "https://")):
            image_url = first
        else:
            image_url = _to_data_uri(first)

    model_lower = _FAL_MODEL.lower()

    # ---------- MiniMax Hailuo 02 ----------
    if "hailuo" in model_lower:
        # Hailuo 02 standard only accepts duration=6 or 10
        duration = 6 if call.duration_s <= 6 else 10
        args: dict = {
            "prompt": call.prompt,
            "duration": duration,
            "resolution": "768P",
            "prompt_optimizer": True,  # MiniMax has a built-in prompt optimizer; let it polish the prompt
        }
        if image_url:
            args["image_url"] = image_url
        # Hailuo doesn't support aspect_ratio (it follows the reference image's ratio) — omit it
        return args

    # ---------- Bytedance Seedance (v1/v2, pro/fast) ----------
    if "seedance" in model_lower:
        args = {
            "prompt": call.prompt,
            "aspect_ratio": call.aspect_ratio,
            "resolution": "1080p",
            # Turn off Seedance 2.0's built-in audio — we use ElevenLabs, and this also avoids audio content policy false positives
            "generate_audio": False,
        }
        # Seedance duration is a string "5" / "10" (v1) or an int (v2). A string has the best compatibility.
        args["duration"] = "5" if call.duration_s <= 5 else "10"
        if image_url:
            args["image_url"] = image_url
        if call.seed is not None:
            args["seed"] = call.seed
        return args

    # ---------- Fallback (other models) ----------
    args = {"prompt": call.prompt}
    if image_url:
        args["image_url"] = image_url
    if call.duration_s:
        args["duration"] = call.duration_s
    return args


# Keep the old name for backward compatibility
_seedance_args = _build_args


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest


def _resolve_image_url(call: SceneAPICall) -> str | None:
    """First reference image as an http(s) URL or a base64 data URI (local files)."""
    if not call.reference_image_uris:
        return None
    first = call.reference_image_uris[0]
    if first.startswith(("http://", "https://")):
        return first
    return _to_data_uri(first)


def _generate_one_volcano(call: SceneAPICall, run_id: str, on_progress=None) -> tuple[str, str]:
    """Generate one scene clip via Volcengine Ark (Seedance 2.0).

    Async API: POST a task → poll until succeeded → download the (24h-expiring) video URL.
    """
    api_key = video_model.ARK_API_KEY
    if not api_key:
        raise RuntimeError("ARK_API_KEY not set (required for VIDEO_PROVIDER=volcano)")
    base = video_model.ARK_BASE_URL.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    content: list[dict] = [{"type": "text", "text": call.prompt}]
    img = _resolve_image_url(call)
    if img:
        image_block: dict = {"type": "image_url", "image_url": {"url": img}}
        if video_model.VOLCANO_IMAGE_ROLE:
            image_block["role"] = video_model.VOLCANO_IMAGE_ROLE
        content.append(image_block)

    # Seedance 2.0 accepts a continuous 4-15s duration (our scenes are 3-10s → clamp to >=4).
    duration = max(4, min(15, int(round(call.duration_s))))
    body: dict = {
        "model": video_model.VOLCANO_VIDEO_MODEL,
        "content": content,
        "ratio": call.aspect_ratio,            # "9:16"
        "resolution": video_model.VOLCANO_RESOLUTION,
        "duration": duration,
        "generate_audio": False,               # we add VO via ElevenLabs
        "watermark": False,
    }
    if call.seed is not None:
        body["seed"] = call.seed

    if on_progress:
        on_progress(call.scene_id, "submitting")
    resp = requests.post(f"{base}/contents/generations/tasks", headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    task_id = resp.json().get("id")
    if not task_id:
        raise RuntimeError(f"Volcano create-task returned no id: {resp.text[:300]}")

    if on_progress:
        on_progress(call.scene_id, "running")

    # Poll until terminal state (10 min budget, 5s interval).
    poll_url = f"{base}/contents/generations/tasks/{task_id}"
    deadline = time.time() + 600
    video_url = None
    while True:
        if time.time() > deadline:
            raise RuntimeError(f"Volcano task {task_id} timed out after 10 min")
        time.sleep(5)
        pr = requests.get(poll_url, headers=headers, timeout=30)
        pr.raise_for_status()
        data = pr.json()
        status = data.get("status")
        if status == "succeeded":
            video_url = (data.get("content") or {}).get("video_url")
            if not video_url:
                raise RuntimeError(f"Volcano task {task_id} succeeded but no video_url: {data}")
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Volcano task {task_id} {status}: {data.get('error') or data}")
        # queued / running → keep polling

    dest = _CLIPS_ROOT / run_id / f"{call.scene_id}.mp4"
    _download(video_url, dest)
    if on_progress:
        on_progress(call.scene_id, "done")
    return call.scene_id, str(dest)


def _generate_one(call: SceneAPICall, run_id: str, on_progress=None) -> tuple[str, str]:
    """Generate one scene clip. Returns (scene_id, mp4_path). Dispatches by provider."""
    if video_model.VIDEO_PROVIDER == "volcano":
        return _generate_one_volcano(call, run_id, on_progress)

    if on_progress:
        on_progress(call.scene_id, "submitting")

    args = _build_args(call)
    handler = fal_client.submit(_FAL_MODEL, arguments=args)

    if on_progress:
        on_progress(call.scene_id, "running")

    result = handler.get()  # blocks until done
    # Result shape: {"video": {"url": "https://..."}} per fal Seedance contract
    video_url = result.get("video", {}).get("url") if isinstance(result.get("video"), dict) else None
    if not video_url:
        # Some fal endpoints return {"url": "..."} flat
        video_url = result.get("url")
    if not video_url:
        raise RuntimeError(f"Seedance returned no video URL: {result}")

    dest = _CLIPS_ROOT / run_id / f"{call.scene_id}.mp4"
    _download(video_url, dest)

    if on_progress:
        on_progress(call.scene_id, "done")

    return call.scene_id, str(dest)


def run_clip_gen(
    director_output: DirectorOutput,
    run_id: str,
    max_parallel: int = 6,
    on_progress=None,
) -> dict[str, str]:
    """
    Run all scenes in parallel via ThreadPoolExecutor.

    Returns: {scene_id: mp4_path} for ALL scenes on full success.
    On partial failure raises ClipGenError carrying the successful clips (so the caller
    can keep them) plus the per-scene errors.
    """
    results: dict[str, str] = {}
    errors: list[tuple[str, Exception]] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {
            ex.submit(_generate_one, call, run_id, on_progress): call.scene_id
            for call in director_output.api_calls
        }
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                _, path = fut.result()
                results[sid] = path
            except Exception as e:
                errors.append((sid, e))

    if errors:
        raise ClipGenError(results, errors)

    return results


if __name__ == "__main__":
    # Manual smoke test stub
    print(f"Using fal model: {_FAL_MODEL}")
    print("Run via run_clip_gen(director_output, run_id) from orchestrator.")
