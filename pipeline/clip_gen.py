"""
Clip Generation runner.

并行调 fal.ai Seedance 2.0 Pro image-to-video, 把每个 scene 的 SceneAPICall 转成实际的 mp4 文件.

设计要点:
- 用 ThreadPoolExecutor 并行 (Streamlit 是 sync 环境, 不引 asyncio)
- 单 clip 失败 → 整个 stage 失败, 抛到 retry router 处理
- 每个 clip 下载到 data/clips/<run_id>/<scene_id>.mp4
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


# 单一事实源: 实际运行的 fal 模型 (见 utils/video_model.py)
_FAL_MODEL = video_model.FAL_VIDEO_MODEL
_CLIPS_ROOT = Path(__file__).parent.parent / "data" / "clips"


class ClipGenError(RuntimeError):
    """部分 clip 生成失败时抛出. 关键: 把已成功的 clip 一并带出来, 让 caller 保留它们,

    用户只需单独重生失败的 scene, 不必把整批 (每个 ~$1) 重跑一遍. 这是省钱的核心.
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

    我们用 data URI 而不是 fal_client.upload_file() — 因为 upload_file 走 fal storage
    需要 storage_write 权限, 而很多 fal key (尤其 free tier) 没开. data URI 直接塞进
    请求 body, 不经过 fal storage, 任何 key 都能用.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_args(call: SceneAPICall) -> dict:
    """Build the input dict — schema differs by model family."""
    # 拿 first reference image, encode 成 data URI 避免 fal storage 权限问题
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
        # Hailuo 02 standard 只接受 duration=6 或 10
        duration = 6 if call.duration_s <= 6 else 10
        args: dict = {
            "prompt": call.prompt,
            "duration": duration,
            "resolution": "768P",
            "prompt_optimizer": True,  # MiniMax 自带 prompt 优化器, 让它帮忙润色
        }
        if image_url:
            args["image_url"] = image_url
        # Hailuo 不支持 aspect_ratio (沿用 reference image 的比例) — 不传
        return args

    # ---------- Bytedance Seedance (v1/v2, pro/fast) ----------
    if "seedance" in model_lower:
        args = {
            "prompt": call.prompt,
            "aspect_ratio": call.aspect_ratio,
            "resolution": "1080p",
            # 关掉 Seedance 2.0 自带音频 — 我们用 ElevenLabs, 同时避免 audio content policy 误报
            "generate_audio": False,
        }
        # Seedance duration 是 string "5" / "10" (v1) 或 int (v2). 用 string 兼容性最好.
        args["duration"] = "5" if call.duration_s <= 5 else "10"
        if image_url:
            args["image_url"] = image_url
        if call.seed is not None:
            args["seed"] = call.seed
        return args

    # ---------- Fallback (其他模型) ----------
    args = {"prompt": call.prompt}
    if image_url:
        args["image_url"] = image_url
    if call.duration_s:
        args["duration"] = call.duration_s
    return args


# 保留旧名字做向后兼容
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


def _generate_one(call: SceneAPICall, run_id: str, on_progress=None) -> tuple[str, str]:
    """Generate one scene clip. Returns (scene_id, mp4_path)."""
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
