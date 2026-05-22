"""
Vizzy Studio — non-linear, editable workflow UI.

Major shifts vs prototype v1 (linear wizard):
  - Sidebar: clickable stage navigation for current run + past runs list
  - Main area: each stage gets its own dedicated view (Brand, Storyboard, References,
    Director Prompts, Clips, Final, QA, Trace)
  - Every stage is EDITABLE — change USP, scene roles, prompts, refs — and re-run
    downstream stages cheaply
  - Per-scene clip regeneration (no need to re-spend on the whole batch)
  - Past runs persisted via data/traces/; click in sidebar to reload any prior run
  - Dark Seedance-inspired theme via .streamlit/config.toml
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# override=True so .env wins over stale/empty vars exported in the shell
load_dotenv(override=True)

from schemas import (
    PipelineState,
    Scene,
    Storyboard,
    BrandUnderstanding,
    DirectorOutput,
    SceneAPICall,
)
from pipeline import orchestrator
from pipeline import clip_jobs
from utils import video_model
from utils.trace import save_trace
from utils.runs import list_runs, load_state, delete_run, set_run_label


def _llm_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _ark_region() -> str:
    """Short region hint from the Ark base URL (e.g. 'cn-beijing')."""
    host = video_model.ARK_BASE_URL.split("//")[-1].split("/")[0]
    return host.replace("ark.", "").replace(".volces.com", "").replace(".bytepluses.com", " (byteplus)")

st.set_page_config(
    page_title="Vizzy Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

REFS_DIR = Path(__file__).parent / "data" / "refs"
TRACE_ROOT = Path(__file__).parent / "data" / "traces"

SCENE_ROLES = ["hook", "problem", "product_reveal", "science", "social_proof",
               "comparison", "demo", "cta"]


# ---------- Session bootstrap ----------
def init_session():
    defaults = {
        "state": None,                  # PipelineState | None
        "current_view": "input",        # which stage view is active
        "log": [],
        "runs_index": None,             # cached list[dict] from list_runs()
        "last_decision": None,          # most recent retry router decision
        "confirm_delete": None,         # run_id pending delete confirmation
        "editing_label": None,          # run_id pending rename
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log(msg: str):
    st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def refresh_runs():
    st.session_state.runs_index = list_runs()


def has_stage(stage: str) -> bool:
    """Has the current state advanced through this stage?"""
    s: PipelineState = st.session_state.state
    if not s:
        return False
    if stage == "brand":       return s.brand is not None
    if stage == "storyboard":  return s.storyboard is not None
    if stage == "references":  return len(s.reference_image_uris) > 0
    if stage == "director":    return s.director_output is not None
    if stage == "clips":       return len(s.clip_paths) > 0
    if stage == "final":       return s.final_video_path is not None and os.path.exists(s.final_video_path)
    if stage == "qa":          return s.qa_report is not None
    return False


def goto(view: str):
    st.session_state.current_view = view


def custom_css():
    """Subtle Seedance-inspired polish on top of the dark theme."""
    st.markdown("""
    <style>
        /* Tighten sidebar */
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] button {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        /* Main heading rhythm */
        .main h1 {
            padding-top: 0.2rem;
            margin-bottom: 0.5rem;
        }
        /* Cards */
        .vizzy-card {
            background: #16161d;
            border: 1px solid #2a2a32;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.6rem;
        }
        .vizzy-pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: #1f1f29;
            border: 1px solid #2a2a32;
            font-size: 0.78rem;
            color: #94949e;
            margin-right: 0.4rem;
        }
        /* Hide Streamlit chrome */
        #MainMenu, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ---------- Init ----------
init_session()
if st.session_state.runs_index is None:
    refresh_runs()
custom_css()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🎬 Vizzy Studio")
    st.caption("Data-driven paid-social video agent")
    st.markdown("---")

    state: PipelineState = st.session_state.state

    if state:
        brand_name = state.brand.name if state.brand else "(no brand yet)"
        st.markdown(f"**Active run**  \n`{state.run_id}`  \n*{brand_name}*")

        nav_items = [
            ("input",      "🌐 URL Input",        True),
            ("brand",      "🏷️  Brand",           has_stage("brand")),
            ("storyboard", "📋 Storyboard",       has_stage("storyboard")),
            ("references", "🖼️  References",      True),
            ("director",   "🎯 Director Prompts", has_stage("director")),
            ("clips",      "🎞️  Clips",           has_stage("clips") or has_stage("director")),
            ("final",      "🎬 Final Video",      has_stage("final")),
            ("qa",         "✅ QA Report",        has_stage("qa")),
            ("trace",      "📜 Trace",            True),
        ]
        for key, label, enabled in nav_items:
            active = st.session_state.current_view == key
            prefix = "→ " if active else "   "
            if st.button(prefix + label, key=f"nav_{key}", disabled=not enabled,
                         use_container_width=True):
                goto(key)
                st.rerun()

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ New run", use_container_width=True):
                st.session_state.state = None
                st.session_state.last_decision = None
                goto("input")
                st.rerun()
        with c2:
            if st.button("🔄 Refresh", use_container_width=True):
                refresh_runs()
                st.rerun()
    else:
        st.caption("No active run.  Start by entering a URL.")
        if st.button("🌐 New URL run", type="primary", use_container_width=True):
            goto("input")
            st.rerun()

    st.markdown("---")
    st.markdown("**📦 Past runs**")
    runs = st.session_state.runs_index or []
    if not runs:
        st.caption("(none yet)")
    else:
        for run in runs[:20]:
            run_id = run["run_id"]
            display_name = run.get("label") or run.get("brand_name") or "—"
            label_id = run_id[-9:]
            mark = "🎬" if run["has_final"] else "·"
            btn_label = f"{mark} {display_name}  {label_id}"

            lc, ec, dc = st.columns([5, 1, 1])
            with lc:
                if st.button(btn_label, key=f"loadrun_{run_id}",
                             use_container_width=True):
                    loaded = load_state(run_id)
                    if loaded:
                        st.session_state.state = loaded
                        # Jump to most-advanced stage
                        for v in ("qa", "final", "clips", "director",
                                  "references", "storyboard", "brand"):
                            if has_stage(v):
                                goto(v)
                                break
                        st.rerun()
            with ec:
                if st.button("✏️", key=f"askrename_{run_id}",
                             help="Rename this run",
                             use_container_width=True):
                    st.session_state.editing_label = run_id
                    st.session_state.confirm_delete = None
                    st.rerun()
            with dc:
                if st.button("🗑", key=f"askdel_{run_id}",
                             help="Delete this run and all its files",
                             use_container_width=True):
                    st.session_state.confirm_delete = run_id
                    st.session_state.editing_label = None
                    st.rerun()

            # Inline rename editor for the run being renamed
            if st.session_state.editing_label == run_id:
                new_label = st.text_input(
                    "New name", value=run.get("label") or run.get("brand_name") or "",
                    key=f"labelinput_{run_id}", placeholder="e.g. Goli hero cut v2",
                    label_visibility="collapsed",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("💾 Save", key=f"savelabel_{run_id}",
                                 type="primary", use_container_width=True):
                        try:
                            set_run_label(run_id, new_label)
                            log(f"Renamed run {label_id} → {new_label.strip() or '(cleared)'}")
                            st.session_state.editing_label = None
                            refresh_runs()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Rename failed: {e}")
                with rc2:
                    if st.button("Cancel", key=f"cancelrename_{run_id}",
                                 use_container_width=True):
                        st.session_state.editing_label = None
                        st.rerun()

            # Confirmation: only expand the confirm bar for the run pending deletion
            if st.session_state.confirm_delete == run_id:
                st.warning(f"Delete `{label_id}`? This permanently removes trace/clips/refs/voiceover/final and cannot be undone.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ Confirm delete", key=f"cfdel_{run_id}",
                                 type="primary", use_container_width=True):
                        try:
                            removed = delete_run(run_id)
                            log(f"Deleted run {run_id} ({len(removed)} path(s))")
                            # If we deleted the current active run, clear it too
                            if state and state.run_id == run_id:
                                st.session_state.state = None
                                st.session_state.last_decision = None
                                goto("input")
                            st.session_state.confirm_delete = None
                            refresh_runs()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
                with cc2:
                    if st.button("Cancel", key=f"cancel_{run_id}",
                                 use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()

    if st.session_state.log:
        st.markdown("---")
        with st.expander("📋 Log", expanded=False):
            for line in st.session_state.log[-15:]:
                st.text(line)


# ---------- Main content area ----------
state = st.session_state.state
view = st.session_state.current_view


def _section_header(title: str, sub: str = ""):
    st.markdown(f"# {title}")
    if sub:
        st.caption(sub)


def render_run_meta(items: list[tuple[str, str]]):
    """Compact runtime-metadata strip (pills) so each stage shows what actually ran —
    model, provider, resolution, cost — instead of looking like a black box."""
    items = [(k, v) for k, v in items if v]
    if not items:
        return
    pills = " ".join(
        f"<span class='vizzy-pill'>{k} · <b>{v}</b></span>" for k, v in items
    )
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


_CLIP_STATUS = {"queued": "⏳ queued", "submitting": "📤 submitting",
                "running": "🎬 generating", "done": "✅ done", "failed": "❌ failed"}


def handle_clip_job(state: PipelineState, scene_ids: list[str]) -> None:
    """Render live per-scene progress for an async clip job (if one exists for this run).

    While running: poll ~every 1.2s via st.rerun(). On completion: commit results to state,
    persist the clip_gen trace, and (on full success) jump to Clips. If some scenes failed,
    fall through so the user can regenerate just those. Returns immediately if no job.
    """
    job = clip_jobs.snapshot(state.run_id)
    if not job:
        return

    total = job["total"]
    done_n = len(job["results"])
    scene_fail = [s for s in job["errors"] if s != "_"]
    st.markdown("### 🎬 Generating clips")
    st.progress(
        min(1.0, (done_n + len(scene_fail)) / max(1, total)),
        text=f"{done_n}/{total} done" + (f" · {len(scene_fail)} failed" if scene_fail else ""),
    )
    for sid in scene_ids:
        if sid in job["results"]:
            stt = "done"
        elif sid in job["errors"]:
            stt = "failed"
        else:
            stt = job["status"].get(sid, "queued")
        st.markdown(f"{_CLIP_STATUS.get(stt, stt)} · **{sid}**")
        if sid in job["errors"]:
            st.caption(job["errors"][sid][:200])
    if "_" in job["errors"]:
        st.error(f"Generation crashed: {job['errors']['_']}")

    if not job["done"]:
        time.sleep(1.2)
        st.rerun()  # raises → stops this run; the poll loop continues on the next

    # Job finished — commit results and persist (this also writes the clip_gen trace that
    # the button path used to skip, so reloads recover correctly).
    state.clip_paths.update(job["results"])
    state.final_video_path = None
    save_trace(state.run_id, "clip_gen",
               {"clip_paths": state.clip_paths, "errors": dict(job["errors"])})
    clip_jobs.clear(state.run_id)
    refresh_runs()
    if scene_fail:
        st.error(f"{len(scene_fail)} scene(s) failed — successful clips kept. "
                 f"Regenerate the failed ones below.")
        log(f"Clip job done: {done_n} ok, {len(scene_fail)} failed")
    else:
        log(f"Clip job done: {done_n} clips")
        goto("clips")
        st.rerun()


# === Input ===
if view == "input" or state is None:
    _section_header("🎬 Vizzy Studio",
                    "Give Vizzy a brand URL or a text brief — it will analyze the brand, "
                    "write a storyboard, and produce a 9:16 paid-social video.")

    url_tab, text_tab = st.tabs(["🌐 From URL", "📝 From text"])

    url = None
    brand_text = None
    run_btn = False
    with url_tab:
        url = st.text_input(
            "Brand URL",
            value="",
            placeholder="https://yourbrand.com/product",
        )
        if st.button("Analyze brand →", type="primary",
                     use_container_width=False, key="run_from_url"):
            run_btn = True
    with text_tab:
        brand_text = st.text_area(
            "Brand / product brief",
            height=200,
            placeholder=("Paste anything that describes the brand — product copy, a description, "
                         "key benefits, audience, tone. Useful when a site blocks scraping or you "
                         "just have notes."),
        )
        if st.button("Analyze brief →", type="primary",
                     use_container_width=False, key="run_from_text"):
            if not (brand_text and brand_text.strip()):
                st.warning("Enter some brand text first.")
            else:
                run_btn = "text"

    if run_btn:
        is_text = run_btn == "text"
        spinner_msg = ("Analyzing brand brief (Strategist ~30s)…" if is_text
                       else "Fetching URL and analyzing brand (Strategist ~30s)…")
        with st.spinner(spinner_msg):
            try:
                if is_text:
                    ns = orchestrator.new_pipeline(brand_text=brand_text.strip())
                else:
                    ns = orchestrator.new_pipeline(brand_url=url)
                ns = orchestrator.stage_strategist(ns, reference_count=0)
                st.session_state.state = ns
                log(f"Strategist complete ({'text' if is_text else 'url'} mode)")
                refresh_runs()
                goto("brand")
                st.rerun()
            except Exception as e:
                st.error(f"Strategist failed: {e}")


# === Brand ===
elif view == "brand":
    _section_header("🏷️ Brand Understanding",
                    "Edit any field; downstream stages (storyboard, director, clips) will be re-runnable.")
    render_run_meta([
        ("agent", "Strategist"),
        ("LLM", _llm_model()),
        ("input", "text" if state.brand_text else "URL"),
    ])

    b = state.brand
    c1, c2 = st.columns(2)
    with c1:
        new_name      = st.text_input("Brand name",        value=b.name)
        new_usp       = st.text_area("USP",                value=b.usp, height=80)
        new_audience  = st.text_area("Target audience",    value=b.target_audience, height=60)
    with c2:
        new_tone      = st.text_input("Tone (comma-sep tags)",     value=", ".join(b.tone))
        new_palette   = st.text_input("Palette (hex, comma-sep)",  value=", ".join(b.palette))
        new_keywords  = st.text_input("Product visual keywords",   value=", ".join(b.product_visual_keywords))
        new_forbidden = st.text_input("Forbidden claims",          value=", ".join(b.forbidden_claims))

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        if st.button("💾 Save edits", use_container_width=True):
            state.brand = BrandUnderstanding(
                name=new_name,
                usp=new_usp,
                target_audience=new_audience,
                tone=[t.strip() for t in new_tone.split(",") if t.strip()],
                palette=[p.strip() for p in new_palette.split(",") if p.strip()],
                product_visual_keywords=[k.strip() for k in new_keywords.split(",") if k.strip()],
                forbidden_claims=[c.strip() for c in new_forbidden.split(",") if c.strip()],
            )
            log("Brand edited")
            st.success("Saved")
            st.rerun()
    with c2:
        if st.button("🔄 Regen storyboard from this brand", use_container_width=True):
            with st.spinner("Strategist regenerating storyboard…"):
                try:
                    from agents.strategist import write_storyboard
                    state.storyboard = write_storyboard(
                        state.brand,
                        reference_count=len(state.reference_image_uris),
                    )
                    # Invalidate downstream
                    state.director_output = None
                    state.clip_paths = {}
                    state.final_video_path = None
                    state.qa_report = None
                    log("Storyboard regenerated; downstream invalidated")
                    goto("storyboard")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")


# === Storyboard ===
elif view == "storyboard":
    sb = state.storyboard
    total = sum(s.duration_s for s in sb.scenes)
    _section_header("📋 Storyboard",
                    f"{len(sb.scenes)} scenes · {total}s total · Schema B (LLM-picked role sequence)")
    render_run_meta([
        ("agent", "Strategist"),
        ("LLM", _llm_model()),
        ("scenes", str(len(sb.scenes))),
        ("total", f"{total}s"),
    ])

    st.info(f"💡 **Why this structure**: {sb.narrative_rationale}")

    for i, scene in enumerate(sb.scenes):
        with st.expander(f"**{scene.id}** · {scene.role} · {scene.duration_s}s", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                new_visual = st.text_area(
                    "Visual description",
                    value=scene.visual_description,
                    height=80, key=f"vis_{scene.id}",
                )
                new_vo = st.text_area(
                    "Voiceover",
                    value=scene.voiceover,
                    height=60, key=f"vo_{scene.id}",
                )
                new_purpose = st.text_input(
                    "Purpose",
                    value=scene.purpose, key=f"pur_{scene.id}",
                )
            with c2:
                new_role = st.selectbox(
                    "Role", SCENE_ROLES,
                    index=SCENE_ROLES.index(scene.role) if scene.role in SCENE_ROLES else 0,
                    key=f"role_{scene.id}",
                )
                new_dur = st.number_input(
                    "Duration (s)",
                    min_value=3, max_value=10,
                    value=scene.duration_s, key=f"dur_{scene.id}",
                )

            sa, sb_, sc = st.columns(3)
            with sa:
                if st.button("💾 Save", key=f"save_{scene.id}"):
                    scene.visual_description = new_visual
                    scene.voiceover = new_vo
                    scene.purpose = new_purpose
                    scene.role = new_role
                    scene.duration_s = new_dur
                    # Invalidate this scene's clip + the assembled video
                    state.director_output = None
                    state.clip_paths.pop(scene.id, None)
                    state.final_video_path = None
                    state.qa_report = None
                    log(f"Scene {scene.id} edited")
                    st.success("Saved")
                    st.rerun()
            with sb_:
                if st.button("🗑️ Delete", key=f"del_{scene.id}"):
                    sb.scenes = [s for s in sb.scenes if s.id != scene.id]
                    state.director_output = None
                    state.clip_paths.pop(scene.id, None)
                    state.final_video_path = None
                    log(f"Scene {scene.id} deleted")
                    st.rerun()

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Continue to References →", type="primary",
                     use_container_width=True):
            goto("references")
            st.rerun()


# === References ===
elif view == "references":
    _section_header("🖼️ Reference Images",
                    "Upload 1-9 images. The first one anchors Hailuo/Seedance's visual style "
                    "via image-to-video. A.1 path: pure user upload (no agent selection).")

    if state.reference_image_uris:
        st.markdown(f"**{len(state.reference_image_uris)} reference(s) loaded**")
        cols = st.columns(min(len(state.reference_image_uris), 4))
        for i, uri in enumerate(state.reference_image_uris):
            with cols[i % 4]:
                if os.path.exists(uri):
                    st.image(uri, use_container_width=True)
                    st.caption(Path(uri).name)
                    if st.button("🗑️ Remove", key=f"rmref_{i}",
                                 use_container_width=True):
                        state.reference_image_uris.pop(i)
                        state.clip_paths = {}
                        state.final_video_path = None
                        log(f"Reference removed: {uri}")
                        st.rerun()

    st.markdown("---")
    uploaded = st.file_uploader(
        "Add reference images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="ref_uploader",
    )
    if uploaded:
        if st.button("Save uploads", use_container_width=False):
            run_refs_dir = REFS_DIR / state.run_id
            run_refs_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                dest = run_refs_dir / f.name
                with open(dest, "wb") as out:
                    out.write(f.read())
                if str(dest) not in state.reference_image_uris:
                    state.reference_image_uris.append(str(dest))
            state.clip_paths = {}
            state.director_output = None
            log(f"Added {len(uploaded)} references")
            st.rerun()

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        ready = len(state.reference_image_uris) > 0
        if st.button("Run Director →", type="primary",
                     disabled=not ready, use_container_width=True):
            with st.spinner("Director generating Seedance/Hailuo prompts…"):
                try:
                    st.session_state.state = orchestrator.stage_director(state)
                    log("Director complete")
                    goto("director")
                    st.rerun()
                except Exception as e:
                    st.error(f"Director failed: {e}")


# === Director Prompts ===
elif view == "director":
    if not has_stage("director"):
        st.warning("Director hasn't run yet for this run.  Click below to generate prompts.")
        if st.button("Run Director →", type="primary"):
            with st.spinner("Director generating prompts…"):
                try:
                    st.session_state.state = orchestrator.stage_director(state)
                    log("Director complete")
                    st.rerun()
                except Exception as e:
                    st.error(f"Director failed: {e}")
        st.stop()

    do = state.director_output
    _section_header("🎯 Director — Per-Scene Prompts",
                    "Each scene's storyboard beat translated into a video-gen prompt.")
    render_run_meta([
        ("prompt LLM", _llm_model()),
        ("video provider", video_model.VIDEO_PROVIDER),
        ("video model", do.api_calls[0].model if do.api_calls else video_model.model_label()),
        ("est. cost", f"${do.estimated_cost_usd:.2f}"),
    ])

    # Live progress if an async clip job is running/just-finished for this run.
    handle_clip_job(state, [c.scene_id for c in do.api_calls])

    for call in do.api_calls:
        with st.expander(f"**{call.scene_id}** · {call.duration_s}s "
                         f"{'· ✓ clip ready' if call.scene_id in state.clip_paths else '· ⌛ no clip yet'}",
                         expanded=False):
            new_prompt = st.text_area(
                "Prompt",
                value=call.prompt,
                height=150,
                key=f"prompt_{call.scene_id}",
            )
            new_neg = st.text_input(
                "Negative prompt",
                value=call.negative_prompt,
                key=f"neg_{call.scene_id}",
            )

            cc1, cc2, cc3 = st.columns([1, 1, 1])
            with cc1:
                if st.button("💾 Save edits", key=f"savedir_{call.scene_id}"):
                    call.prompt = new_prompt
                    call.negative_prompt = new_neg
                    state.clip_paths.pop(call.scene_id, None)
                    state.final_video_path = None
                    log(f"Prompt {call.scene_id} edited")
                    st.success("Saved — clip invalidated")
                    st.rerun()
            with cc2:
                if st.button("🎞️ Regen this clip", key=f"regen_{call.scene_id}"):
                    with st.spinner(f"Regenerating {call.scene_id} (~60-120s)…"):
                        try:
                            from pipeline.clip_gen import _generate_one
                            sid, path = _generate_one(call, state.run_id)
                            state.clip_paths[sid] = path
                            state.final_video_path = None
                            state.qa_report = None
                            log(f"Clip {sid} regenerated")
                            st.success(f"{sid} regenerated")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
            with cc3:
                if call.scene_id in state.clip_paths and \
                        os.path.exists(state.clip_paths[call.scene_id]):
                    st.video(state.clip_paths[call.scene_id])

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        missing = [c.scene_id for c in do.api_calls if c.scene_id not in state.clip_paths]
        if missing:
            if st.button(f"Generate {len(missing)} missing clip(s) in parallel →",
                         type="primary", use_container_width=True):
                missing_calls = [c for c in do.api_calls if c.scene_id in missing]
                sub_output = DirectorOutput(api_calls=missing_calls, estimated_cost_usd=0)
                # Launch async — handle_clip_job (top of this view) renders live progress.
                clip_jobs.start(state.run_id, sub_output)
                log(f"Started clip job for {len(missing_calls)} scene(s)")
                st.rerun()
        else:
            if st.button("All clips ready · Continue to Clips →",
                         type="primary", use_container_width=True):
                goto("clips")
                st.rerun()


# === Clips ===
elif view == "clips":
    scenes = state.storyboard.scenes if state.storyboard else []
    n_ready = sum(1 for s in scenes if s.id in state.clip_paths)
    _section_header("🎞️ Generated Clips",
                    f"{n_ready} / {len(scenes)} clips generated")
    _is_volcano = video_model.VIDEO_PROVIDER == "volcano"
    render_run_meta([
        ("provider", video_model.VIDEO_PROVIDER),
        ("model", video_model.active_model_id()),
        ("resolution", video_model.VOLCANO_RESOLUTION if _is_volcano else "1080p"),
        ("region", _ark_region() if _is_volcano else "fal.ai"),
    ])

    # Show live progress here too, so navigating to Clips mid-generation isn't a black box.
    handle_clip_job(state, [s.id for s in scenes])

    cpr = 3
    for row_start in range(0, len(scenes), cpr):
        cols = st.columns(cpr)
        for col_idx, scene in enumerate(scenes[row_start:row_start + cpr]):
            with cols[col_idx]:
                st.markdown(f"**{scene.id}**  ·  {scene.role}  ·  {scene.duration_s}s")
                cp = state.clip_paths.get(scene.id)
                if cp and os.path.exists(cp):
                    st.video(cp)
                else:
                    st.warning("Not generated")

                if st.button(f"🔄 Regen {scene.id}",
                             key=f"regclips_{scene.id}",
                             use_container_width=True):
                    call = None
                    if state.director_output:
                        call = next(
                            (c for c in state.director_output.api_calls if c.scene_id == scene.id),
                            None,
                        )
                    if not call:
                        st.error("No Director prompt — run Director first")
                    else:
                        with st.spinner(f"Regen {scene.id}…"):
                            try:
                                from pipeline.clip_gen import _generate_one
                                sid, path = _generate_one(call, state.run_id)
                                state.clip_paths[sid] = path
                                state.final_video_path = None
                                state.qa_report = None
                                log(f"Clip {sid} regenerated")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        all_ready = all(s.id in state.clip_paths for s in scenes)
        if st.button("Edit final video →", type="primary",
                     disabled=not all_ready, use_container_width=True):
            with st.spinner("Stitching, adding VO, burning subs (~30-60s)…"):
                try:
                    st.session_state.state = orchestrator.stage_editor(state)
                    log("Editor complete")
                    refresh_runs()
                    goto("final")
                    st.rerun()
                except Exception as e:
                    st.error(f"Editor failed: {e}")


# === Final ===
elif view == "final":
    _section_header("🎬 Final Video", f"Run `{state.run_id}`")
    render_run_meta([
        ("video model", state.director_output.api_calls[0].model
            if state.director_output and state.director_output.api_calls else video_model.model_label()),
        ("VO", os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5")),
        ("voice", os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")),
        ("stitch", "ffmpeg · 1080x1920"),
    ])

    if state.final_video_path and os.path.exists(state.final_video_path):
        st.video(state.final_video_path)
        st.caption(state.final_video_path)
        with open(state.final_video_path, "rb") as f:
            st.download_button(
                "⬇️ Download mp4",
                f, file_name=f"vizzy_{state.run_id}.mp4",
                type="primary",
            )
    else:
        st.warning("No final video yet — run Editor from the Clips tab.")

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("Run QA →", type="primary",
                     disabled=not has_stage("final"),
                     use_container_width=True):
            with st.spinner("Claude vision reviewing frames (~30s)…"):
                try:
                    st.session_state.state, decision = orchestrator.stage_qa(state)
                    st.session_state.last_decision = decision.to_dict()
                    log(f"QA: {decision.action} — {decision.reason}")
                    goto("qa")
                    st.rerun()
                except Exception as e:
                    st.error(f"QA failed: {e}")
    with c2:
        if st.button("🔄 Re-edit", use_container_width=True):
            with st.spinner("Re-editing…"):
                try:
                    st.session_state.state = orchestrator.stage_editor(state)
                    log("Re-edited")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")


# === QA ===
elif view == "qa":
    _section_header("✅ QA Report",
                    "Claude vision frame sampling + spelling + brand consistency + claim compliance checks")
    render_run_meta([
        ("agent", "QA"),
        ("vision model", _llm_model()),
        ("frames sampled", "4"),
    ])

    qa = state.qa_report
    decision = st.session_state.last_decision or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ranking", qa.ranking)
    c2.metric("Spelling",     "✓" if qa.spelling.get("ok") else "✗")
    c3.metric("Brand",        "✓" if qa.brand_consistency.get("ok") else "✗")
    c4.metric("Compliance",   "✓" if qa.claim_compliance.get("ok") else "✗")

    if qa.faults:
        st.markdown("### Faults found")
        for f in qa.faults:
            severity_emoji = {"block": "🔴", "warn": "🟡", "info": "🔵"}.get(f.severity, "⚪")
            st.markdown(f"{severity_emoji} **{f.fault_type}** · scene `{f.scene_id}` · {f.severity}")
            st.caption(f.reason)
    else:
        st.success("✓ No faults detected")

    st.markdown("### Retry router decision")
    if decision:
        action = decision.get("action", "?")
        target = decision.get("retry_from_stage")
        reason = decision.get("reason", "")
        if action == "continue":
            st.success(f"✓ Continue — {reason}")
        elif action == "retry":
            st.info(f"↪ Retry from `{target}` — {reason}")
            pending = state.active_constraints.get(target, {})
            if pending:
                st.markdown("**Strengthened directives that will be injected on re-run:**")
                for v in pending.values():
                    st.caption(f"• {v}")
            if st.button(f"↪ Re-run {target} with directives →", type="primary"):
                try:
                    if target == "strategist":
                        with st.spinner("Strategist re-running with QA directives…"):
                            st.session_state.state = orchestrator.stage_strategist(
                                state, reference_count=len(state.reference_image_uris)
                            )
                        s2 = st.session_state.state
                        s2.director_output = None
                        s2.clip_paths = {}
                        s2.final_video_path = None
                        s2.qa_report = None
                        log("Retry: Strategist re-ran with QA directives; downstream invalidated")
                        goto("storyboard")
                    elif target == "director":
                        with st.spinner("Director re-running with QA directives…"):
                            st.session_state.state = orchestrator.stage_director(state)
                        s2 = st.session_state.state
                        s2.clip_paths = {}
                        s2.final_video_path = None
                        s2.qa_report = None
                        log("Retry: Director re-ran with QA directives; clips invalidated")
                        goto("director")
                    else:
                        goto({"editor": "final", "clip_gen": "clips"}.get(target, "input"))
                    st.rerun()
                except Exception as e:
                    st.error(f"Retry failed: {e}")
        elif action == "halt":
            st.error(f"⏹ Halted — {reason}")

    with st.expander("Raw QA JSON", expanded=False):
        st.json(qa.model_dump())


# === Trace ===
elif view == "trace":
    _section_header("📜 Pipeline Trace",
                    f"Persisted stage-by-stage JSON outputs for run `{state.run_id}`")

    trace_dir = TRACE_ROOT / state.run_id
    if not trace_dir.exists():
        st.warning("No trace files persisted for this run yet.")
    else:
        trace_files = sorted(trace_dir.glob("*.json"))
        if not trace_files:
            st.caption("(empty)")
        for tp in trace_files:
            with st.expander(f"📄 `{tp.name}`", expanded=False):
                try:
                    with open(tp) as f:
                        data = json.load(f)
                    st.json(data)
                except Exception as e:
                    st.error(f"Could not read: {e}")
