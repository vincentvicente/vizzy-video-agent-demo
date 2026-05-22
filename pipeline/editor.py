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
_CLIPS_ROOT = Path(__file__).parent.parent / "data" / "clips"
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


def _build_voice_settings(settings: dict):
    """Build an ElevenLabs VoiceSettings from the user's options, or None if no style knobs set."""
    keys = ("stability", "similarity_boost", "style", "use_speaker_boost")
    if not any(k in settings for k in keys):
        return None
    try:
        from elevenlabs import VoiceSettings
        return VoiceSettings(
            stability=float(settings.get("stability", 0.5)),
            similarity_boost=float(settings.get("similarity_boost", 0.75)),
            style=float(settings.get("style", 0.0)),
            use_speaker_boost=bool(settings.get("use_speaker_boost", True)),
        )
    except Exception:
        return None


def _generate_voiceovers(
    storyboard: Storyboard, run_id: str, max_parallel: int = 4,
    settings: Optional[dict] = None,
) -> dict[str, Path]:
    """One ElevenLabs TTS call per scene's voiceover, run in parallel. Returns {scene_id: mp3 path}.

    VO calls are network-IO bound and independent per scene — parallelizing collapses N
    serial round trips into roughly 1 round trip of wall-clock time.

    settings: optional voiceover options (voice_id, model_id, stability, similarity_boost,
    style, use_speaker_boost). Falls back to the env voice + turbo model.
    """
    settings = settings or {}
    voice_id = settings.get("voice_id") or _VOICE_ID
    model_id = settings.get("model_id") or "eleven_turbo_v2_5"
    voice_settings = _build_voice_settings(settings)

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
    out_dir = _VO_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes_with_vo = [s for s in storyboard.scenes if s.voiceover.strip()]

    def _one(s) -> tuple[str, Path]:
        kwargs = dict(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            text=s.voiceover,
            model_id=model_id,
        )
        if voice_settings is not None:
            kwargs["voice_settings"] = voice_settings
        audio = client.text_to_speech.convert(**kwargs)
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


def _tts_to_file(text: str, dest: Path, settings: Optional[dict] = None) -> Path:
    """One ElevenLabs TTS call → mp3 at dest (honors voice/model/style settings)."""
    settings = settings or {}
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
    kwargs = dict(
        voice_id=settings.get("voice_id") or _VOICE_ID,
        output_format="mp3_44100_128",
        text=text,
        model_id=settings.get("model_id") or "eleven_turbo_v2_5",
    )
    vsobj = _build_voice_settings(settings)
    if vsobj is not None:
        kwargs["voice_settings"] = vsobj
    audio = client.text_to_speech.convert(**kwargs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)
    return dest


def build_scene_preview(
    scene_id: str, voiceover_text: str, clip_path: str, run_id: str,
    settings: Optional[dict] = None,
) -> Path:
    """Mux a single scene's ElevenLabs VO onto its (silent) clip → a preview mp4 WITH sound.

    Lets the Clips view play each clip with its voiceover. Previews live in
    data/clips/<run_id>/previews/ (a subdir, so they're not mistaken for source clips).
    Returns the silent clip path unchanged if there's no voiceover text.
    """
    clip = Path(clip_path)
    if not voiceover_text.strip():
        return clip
    vo = _VO_ROOT / run_id / f"{scene_id}.mp3"
    _tts_to_file(voiceover_text, vo, settings)
    preview = clip.parent / "previews" / f"{scene_id}.mp4"
    preview.parent.mkdir(parents=True, exist_ok=True)
    # Play the FULL voiceover. Don't use -shortest: the clip is often shorter than the spoken
    # line and -shortest would truncate the VO. Instead make the output = max(clip, VO) length —
    # freeze the clip's last frame if the VO outlasts it; pad the audio with silence otherwise.
    try:
        target = max(_ffprobe_duration(clip), _ffprobe_duration(vo))
    except Exception:
        target = 0.0
    cmd = [
        "ffmpeg", "-y", "-i", str(clip), "-i", str(vo),
        "-filter_complex", "[0:v]tpad=stop_mode=clone:stop_duration=3600[v];[1:a]apad[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-c:a", "aac",
    ]
    if target > 0:
        cmd += ["-t", f"{target:.3f}"]
    cmd += [str(preview)]
    _run(cmd)
    return preview


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
    vos: dict[str, Path], storyboard: Storyboard, run_id: str,
    durations: dict[str, float],
) -> Optional[Path]:
    """Build a single audio track with VO aligned to scene start times.

    durations[scene_id] is the scene's EFFECTIVE length (>= VO length), so the VO is padded
    with trailing silence to fill the scene but never trimmed — every line plays in full.
    """
    if not vos:
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"audio_{run_id}_"))
    scene_audio_paths: list[Path] = []
    for s in storyboard.scenes:
        d = durations.get(s.id, float(s.duration_s))
        scene_audio = tmp_dir / f"{s.id}.wav"
        if s.id in vos:
            # Pad VO with trailing silence to the effective scene duration (no trim — d >= VO len)
            cmd = [
                "ffmpeg", "-y", "-i", str(vos[s.id]),
                "-af", f"apad=whole_dur={d:.3f}",
                "-ar", "44100", "-ac", "2",
                str(scene_audio),
            ]
        else:
            # Pure silence for this scene
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", f"{d:.3f}",
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


def _build_srt(storyboard: Storyboard, run_id: str, durations: dict[str, float]) -> Path:
    """Generate SRT subtitle file from storyboard voiceover lines, timed by effective duration."""
    srt_path = _VO_ROOT / run_id / "subtitles.srt"
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    t = 0.0
    for idx, s in enumerate(storyboard.scenes, start=1):
        start = t
        end = t + durations.get(s.id, float(s.duration_s))
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


_VO_TAIL_S = 0.4  # breathing room after a spoken line before the scene cuts


def _effective_durations(storyboard: Storyboard, vos: dict[str, Path]) -> dict[str, float]:
    """Per-scene effective length = max(storyboard duration, VO length + small tail).

    This makes the VOICEOVER drive timing: if a line is longer than its storyboard duration,
    the scene is stretched (clip's last frame frozen) so the line is never cut off.
    """
    eff: dict[str, float] = {}
    for s in storyboard.scenes:
        base = float(s.duration_s)
        vo = vos.get(s.id)
        vo_dur = 0.0
        if vo and Path(vo).exists():
            try:
                vo_dur = _ffprobe_duration(Path(vo))
            except Exception:
                vo_dur = 0.0
        eff[s.id] = max(base, vo_dur + _VO_TAIL_S) if vo_dur > 0 else base
    return eff


def run_editor(
    clip_paths: dict[str, str],
    storyboard: Storyboard,
    run_id: str,
    burn_subtitles: bool = True,
    voiceover_settings: Optional[dict] = None,
) -> Path:
    """
    Main editor entry point. VOICEOVER-DRIVEN: each scene is stretched to fit its full VO line.

    voiceover_settings: optional VO options (voice/model/style) applied to ElevenLabs TTS.
    Returns Path to final mp4 in data/final/<run_id>.mp4
    """
    work = Path(tempfile.mkdtemp(prefix=f"editor_{run_id}_"))

    # Step 1: generate VOs. If TTS fails (e.g. ElevenLabs free tier can't use library
    # voices → 402), degrade to a silent video instead of failing the whole Editor.
    try:
        vos = _generate_voiceovers(storyboard, run_id, settings=voiceover_settings)
    except Exception as e:
        print(f"[editor] WARNING: voiceover generation failed — producing video WITHOUT VO. {e}")
        vos = {}

    # Effective per-scene durations: stretch any scene whose VO outlasts its storyboard length.
    durations = _effective_durations(storyboard, vos)

    # Step 2: normalize clips in parallel to their effective durations (freeze last frame to fill).
    for s in storyboard.scenes:
        if s.id not in clip_paths:
            raise ValueError(f"Editor missing clip for scene {s.id}")

    def _norm_one(s) -> tuple[str, Path]:
        return s.id, _normalize_clip(
            Path(clip_paths[s.id]), durations[s.id], work / f"norm_{s.id}.mp4"
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

    # Steps 4-6: assemble audio + subtitles onto the stitched video.
    final_path = _assemble_final(concat_video, vos, storyboard, run_id, durations, burn_subtitles)
    shutil.rmtree(work, ignore_errors=True)
    return final_path


def _assemble_final(
    concat_video: Path, vos: dict[str, Path], storyboard: Storyboard, run_id: str,
    durations: dict[str, float], burn_subtitles: bool = True,
) -> Path:
    """Build the timeline-aligned audio + subtitles (using effective durations) and mux them
    onto the stitched silent video → final mp4."""
    _FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    final_path = _FINAL_ROOT / f"{run_id}.mp4"
    work = Path(tempfile.mkdtemp(prefix=f"assemble_{run_id}_"))

    combined_audio = _build_combined_audio(vos, storyboard, run_id, durations)
    srt_path = _build_srt(storyboard, run_id, durations) if burn_subtitles else None

    # The ffmpeg subtitles filter crashes when parsing paths that contain spaces — work around it
    # via cwd: copy the SRT into the work dir and reference only the filename in the filter.
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

    if srt_path and burn_subtitles:
        try:
            shutil.copyfile(srt_path, work / "subs.srt")
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
