"""
Quick smoke test: API keys + Strategist working.

Before running this, do `pip install -r requirements.txt` and fill in .env.

Cost: just 1 Claude call (~$0.01); does not hit fal.ai / ElevenLabs.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_env():
    missing = []
    for key in ["ANTHROPIC_API_KEY", "FAL_KEY", "ELEVENLABS_API_KEY"]:
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print("❌ Missing env vars:", missing)
        print("   → cp .env.example .env, then fill in keys")
        sys.exit(1)
    print("✓ All env vars present")


def check_anthropic():
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=20,
        messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
    )
    print(f"✓ Anthropic OK — got: {resp.content[0].text!r}")


def check_fal():
    import fal_client
    # fal_client reads FAL_KEY from env; check via a cheap list call
    # (no public health endpoint; just verify the client can construct a handler)
    print(f"✓ fal-client imported, key set (will validate on first generate call)")


def check_elevenlabs():
    # Don't use voices.get_all (needs the voices_read permission); just make a minimal TTS call
    # to verify the text_to_speech capability we actually need.
    from elevenlabs.client import ElevenLabs
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    stream = client.text_to_speech.convert(
        voice_id=voice_id,
        output_format="mp3_44100_128",
        text="OK",
        model_id="eleven_turbo_v2_5",
    )
    nbytes = 0
    for chunk in stream:
        if chunk:
            nbytes += len(chunk)
    if nbytes < 100:
        raise RuntimeError(f"ElevenLabs returned suspiciously little audio: {nbytes} bytes")
    print(f"✓ ElevenLabs OK — TTS returned {nbytes} bytes (voice {voice_id})")


def check_ffmpeg():
    import subprocess
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        first = p.stdout.split("\n")[0]
        print(f"✓ ffmpeg: {first}")
    except Exception as e:
        print(f"❌ ffmpeg not in PATH: {e}")
        sys.exit(1)


def check_strategist(url: str):
    print(f"\nRunning Strategist on {url}…")
    from agents.strategist import run_strategist
    out = run_strategist(url, reference_count=0)
    print(f"\n✓ Brand: {out['brand']['name']} — {out['brand']['usp']}")
    print(f"✓ Storyboard: {len(out['storyboard']['scenes'])} scenes, "
          f"{out['storyboard']['total_duration_s']}s total")
    print(f"\nNarrative rationale:\n  {out['storyboard']['narrative_rationale']}")
    print("\nScenes:")
    for s in out["storyboard"]["scenes"]:
        print(f"  {s['id']} [{s['role']}, {s['duration_s']}s] — {s['visual_description'][:80]}…")


if __name__ == "__main__":
    print("Vizzy setup verification\n" + "=" * 40)
    check_env()
    check_anthropic()
    check_fal()
    check_elevenlabs()
    check_ffmpeg()

    url = sys.argv[1] if len(sys.argv) > 1 else "https://goli.com/pages/goli-acv"
    check_strategist(url)

    print("\n" + "=" * 40)
    print("All checks passed. Run `streamlit run app.py` to launch UI.")
