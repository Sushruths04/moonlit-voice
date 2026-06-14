"""VoxCPM2 voice cloning and narration — Modal GPU only."""

from __future__ import annotations

import io
import os
import tempfile

MODEL_ID = "openbmb/VoxCPM2"


def clone_and_speak(ref_wav: str, text: str, speed: float = 0.9, mood: str = "", energy: float = 0.45) -> str:
    """Clone the reference voice and synthesize text to a temporary WAV path.

    Always runs on a Modal GPU — never on the local machine.
    *mood* (magical/funny/calming/dreamy) and *energy* (0=calm..1=lively) shape the
    expressive style tags + pacing of the narration.
    """
    if not ref_wav or not os.path.exists(ref_wav):
        raise ValueError("Please provide a prepared voice reference WAV.")

    story_text = (text or "").strip()
    if not story_text:
        raise ValueError("Please provide story text to narrate.")

    return _clone_and_speak_modal(ref_wav, story_text, speed, mood, energy)


def _clone_and_speak_modal(ref_wav: str, text: str, speed: float, mood: str, energy: float) -> str:
    import modal
    import soundfile as sf

    with open(ref_wav, "rb") as ref_file:
        ref_wav_bytes = ref_file.read()

    try:
        synthesize_story = modal.Function.from_name("dreamvoice-tts", "synthesize_story")
        wav_bytes = synthesize_story.remote(ref_wav_bytes, text, speed, mood, energy)
    except Exception as exc:  # noqa: BLE001 - surface a user-actionable app error.
        raise RuntimeError(
            "Modal VoxCPM2 synthesis failed. Confirm `modal deploy modal_app.py` has run "
            f"and Modal credentials are configured. Original error: {exc}"
        ) from exc

    from audio_postprocess import postprocess

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    audio = postprocess(audio, sr)

    fd, out_path = tempfile.mkstemp(prefix="dreamvoice_story_", suffix=".wav")
    os.close(fd)
    sf.write(out_path, audio, sr)
    return out_path
