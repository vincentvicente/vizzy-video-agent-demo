"""
Editor — ffmpeg concat + ElevenLabs VO + subtitle overlay.

Pipeline:
  1. Generate VO per scene via ElevenLabs (1 TTS call per scene)
  2. Normalize each clip → 1080x1920 9:16, trimmed to storyboard duration
  3. Concat clips → silent video track
  4. Build combined audio: VOs aligned to scene start, padded with silence
  5. Generate SRT with VO text per scene
  6. Mux video + audio + burn subtitles → final.mp4
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from elevenlabs.client import ElevenLabs

from schemas import Storyboard


_VO_ROOT = Path(__file__).parent.parent / "data" / "voiceover"
_FINAL_ROOT = Path(__file__).parent.parent / "data" / "final"
_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel default


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, capture output for debugging."""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd[:3])}…):\n"
            f"STDOUT: {proc.stdout[-500:]}\n"
            f"STDERR: {proc.stderr[-500:]}"
        )
    return proc


def _ffprobe_duration(path: Path) -> float:
    """Get media duration in seconds."""
    p = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(p.stdout.strip())


def _generate_voiceovers(
    storyboard: Storyboard, run_id: str, max_parallel: int = 4
) -> dict[str, Path]:
    """One ElevenLabs TTS call per scene's voiceover, run in parallel. Returns {scene_id: mp3 path}.

    VO calls are network-IO bound and independent per scene — parallelizing collapses N
    serial round trips into roughly 1 round trip of wall-clock time.
    """
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
    out_dir = _VO_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes_with_vo = [s for s in storyboard.scenes if s.voiceover.strip()]

    def _one(s) -> tuple[str, Path]:
        audio = client.text_to_speech.convert(
            voice_id=_VOICE_ID,
            output_format="mp3_44100_128",
            text=s.voiceover,
            model_id="eleven_turbo_v2_5",
        )
        path = out_dir / f"{s.id}.mp3"
        with open(path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        return s.id, path

    vos: dict[str, Path] = {}
    if not scenes_with_vo:
        return vos
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = [ex.submit(_one, s) for s in scenes_with_vo]
        for fut in as_completed(futs):
            sid, path = fut.result()
            vos[sid] = path
    return vos


def _normalize_clip(src: Path, target_duration_s: float, dest: Path) -> Path:
    """Scale/pad clip to 1080x1920 9:16 and trim/extend to target duration."""
    # Scale & pad to 1080x1920
    # Trim to exact target_duration_s (if too short, will use last-frame freeze via tpad)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf",
        (
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps=30,"
            f"tpad=stop_mode=clone:stop_duration={target_duration_s}"
        ),
        "-t", str(target_duration_s),
        "-an",  # strip audio
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        str(dest),
    ]
    _run(cmd)
    return dest


def _build_combined_audio(
    vos: dict[str, Path], storyboard: Storyboard, run_id: str
) -> Optional[Path]:
    """Build a single audio track with VO aligned to scene start times."""
    if not vos:
        return None

    # Build per-scene audio: pad each VO with silence to match scene duration
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"audio_{run_id}_"))
    scene_audio_paths: list[Path] = []
    for s in storyboard.scenes:
        scene_audio = tmp_dir / f"{s.id}.wav"
        if s.id in vos:
            # Trim VO to scene duration max, then pad to scene duration with silence
            cmd = [
                "ffmpeg", "-y", "-i", str(vos[s.id]),
                "-af",
                f"apad=whole_dur={s.duration_s},atrim=0:{s.duration_s}",
                "-ar", "44100", "-ac", "2",
                str(scene_audio),
            ]
        else:
            # Pure silence for this scene
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(s.duration_s),
                str(scene_audio),
            ]
        _run(cmd)
        scene_audio_paths.append(scene_audio)

    # Concat
    list_file = tmp_dir / "audio_concat.txt"
    with open(list_file, "w") as f:
        for p in scene_audio_paths:
            f.write(f"file '{p}'\n")
    combined = tmp_dir / "combined.wav"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(combined)
    ])
    return combined


def _build_srt(storyboard: Storyboard, run_id: str) -> Path:
    """Generate SRT subtitle file from storyboard voiceover lines, timed per scene."""
    srt_path = _VO_ROOT / run_id / "subtitles.srt"
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    t = 0.0
    for idx, s in enumerate(storyboard.scenes, start=1):
        start = t
        end = t + s.duration_s
        if s.voiceover.strip():
            lines.append(f"{idx}")
            lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
            lines.append(s.voiceover.strip())
            lines.append("")
        t = end
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_editor(
    clip_paths: dict[str, str],
    storyboard: Storyboard,
    run_id: str,
    burn_subtitles: bool = True,
) -> Path:
    """
    Main editor entry point.

    Returns Path to final mp4 in data/final/<run_id>.mp4
    """
    final_dir = _FINAL_ROOT
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{run_id}.mp4"

    work = Path(tempfile.mkdtemp(prefix=f"editor_{run_id}_"))

    # Step 1: generate VOs. If TTS fails (e.g. ElevenLabs free tier can't use library
    # voices → 402), degrade to a silent video instead of failing the whole Editor.
    # Subtitles still burn (they come from storyboard text, not from TTS).
    try:
        vos = _generate_voiceovers(storyboard, run_id)
    except Exception as e:
        print(f"[editor] WARNING: voiceover generation failed — producing video WITHOUT VO. {e}")
        vos = {}

    # Step 2: normalize clips in parallel (independent ffmpeg subprocesses), then assemble them in storyboard order.
    for s in storyboard.scenes:
        if s.id not in clip_paths:
            raise ValueError(f"Editor missing clip for scene {s.id}")

    def _norm_one(s) -> tuple[str, Path]:
        return s.id, _normalize_clip(
            Path(clip_paths[s.id]), s.duration_s, work / f"norm_{s.id}.mp4"
        )

    norm_by_id: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_norm_one, s) for s in storyboard.scenes]
        for fut in as_completed(futs):
            sid, path = fut.result()
            norm_by_id[sid] = path
    normalized: list[Path] = [norm_by_id[s.id] for s in storyboard.scenes]

    # Step 3: concat video
    list_file = work / "video_concat.txt"
    with open(list_file, "w") as f:
        for p in normalized:
            f.write(f"file '{p}'\n")
    concat_video = work / "concat.mp4"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(concat_video)
    ])

    # Step 4: combined audio
    combined_audio = _build_combined_audio(vos, storyboard, run_id)

    # Step 5: SRT
    srt_path = _build_srt(storyboard, run_id) if burn_subtitles else None

    # Step 6: final mux + (optional) burn subtitles
    # The ffmpeg subtitles filter crashes when parsing paths that contain spaces — work around it via cwd:
    # copy the SRT into the work dir, run ffmpeg with cwd set there, and reference only the filename in the filter.

    def _build_base_cmd() -> list[str]:
        base = ["ffmpeg", "-y", "-i", str(concat_video)]
        if combined_audio:
            base += ["-i", str(combined_audio),
                     "-map", "0:v:0", "-map", "1:a:0",
                     "-c:a", "aac", "-b:a", "192k"]
        else:
            base += ["-an"]
        return base

    encode_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20"]

    subtitle_attempted = False
    if srt_path and burn_subtitles:
        subtitle_attempted = True
        try:
            safe_srt = work / "subs.srt"
            shutil.copyfile(srt_path, safe_srt)
            cmd = _build_base_cmd() + [
                "-vf",
                "subtitles=subs.srt:force_style='Fontsize=20,PrimaryColour=&Hffffff&,BorderStyle=3,Outline=2,Shadow=0,MarginV=160,Alignment=2'",
            ] + encode_args + [str(final_path)]
            proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg subtitle burn failed (will retry without subs):\n"
                    f"STDERR: {proc.stderr[-300:]}"
                )
            print(f"[editor] final video written WITH burned subtitles: {final_path}")
        except RuntimeError as e:
            # Subtitle burn failed → fall back to a version without subtitles instead of killing the whole pipeline.
            # The VO is still present (from combined_audio), so the video is still usable for the demo.
            print(f"[editor] WARNING: {e}")
            print(f"[editor] retrying without subtitle burn — final video will have VO only")
            cmd = _build_base_cmd() + encode_args + [str(final_path)]
            _run(cmd)
            print(f"[editor] final video written WITHOUT subtitles (fallback): {final_path}")
    else:
        cmd = _build_base_cmd() + encode_args + [str(final_path)]
        _run(cmd)

    shutil.rmtree(work, ignore_errors=True)
    return final_path


if __name__ == "__main__":
    print("Editor module loaded. Use run_editor(clip_paths, storyboard, run_id).")
