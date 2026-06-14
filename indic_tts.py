"""Kannada narration in the parent's cloned voice via AI4Bharat IndicF5 (Modal GPU).

IndicF5 clones a voice from a reference clip + its transcript and speaks Kannada
zero-shot. We synthesize sentence-by-sentence (see modal_app) so the narration has
natural prosody instead of a flat, continuous machine tone.
"""

from __future__ import annotations

import io
import os
import tempfile

MODEL_ID = "ai4bharat/IndicF5"


def narrate_kannada(ref_wav: str, ref_text: str, kannada_text: str, mood: str = "", energy: float = 0.45) -> str:
    """Clone the parent's voice and narrate Kannada text. Returns a temp WAV path.

    *ref_text* MUST be the transcript of *ref_wav* (IndicF5 requires it). In the app
    the parent reads a known sentence we display, so we always know it exactly.
    *energy* (0=calm..1=lively) shapes the pacing.
    """
    if not ref_wav or not os.path.exists(ref_wav):
        raise ValueError("Please provide a prepared voice reference WAV.")
    if not (ref_text or "").strip():
        raise ValueError("Reference transcript (ref_text) is required for Kannada cloning.")
    if not (kannada_text or "").strip():
        raise ValueError("Please provide Kannada text to narrate.")

    import modal
    import soundfile as sf

    with open(ref_wav, "rb") as fh:
        ref_bytes = fh.read()

    try:
        fn = modal.Function.from_name("dreamvoice-tts", "synthesize_kannada")
        wav_bytes = fn.remote(ref_bytes, ref_text.strip(), kannada_text.strip(), mood, energy)
    except Exception as exc:  # noqa: BLE001 - surface a user-actionable app error.
        raise RuntimeError(
            "Kannada narration (IndicF5) failed. Confirm `modal deploy modal_app.py` has run. "
            f"Original error: {exc}"
        ) from exc

    from audio_postprocess import postprocess

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    audio = postprocess(audio, sr)

    fd, out_path = tempfile.mkstemp(prefix="dreamvoice_kn_", suffix=".wav")
    os.close(fd)
    sf.write(out_path, audio, sr)
    return out_path
