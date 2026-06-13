"""Audio preparation helpers for DreamVoice reference clips."""

from __future__ import annotations

import os
import tempfile

import librosa
import numpy as np
import soundfile as sf

REFERENCE_SAMPLE_RATE = 16_000
MIN_REFERENCE_SECONDS = 5.0
MAX_REFERENCE_SECONDS = 60.0
MIN_RMS = 0.005


def prepare_reference(path: str) -> str:
    """Clean and validate a voice reference clip for VoxCPM2.

    The returned file is a temporary mono 16 kHz WAV. The caller owns cleanup of
    that returned path after synthesis finishes.
    """
    if not path or not os.path.exists(path):
        raise ValueError("Please record or upload a voice clip first.")

    try:
        audio, _ = librosa.load(path, sr=REFERENCE_SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 - present a friendly UI-safe error.
        raise ValueError("I couldn't read that audio clip. Please try a WAV or MP3 recording.") from exc

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("That voice clip looks empty. Please record 5-60 seconds of clear speech.")

    audio = np.asarray(audio, dtype=np.float32)
    audio, _ = librosa.effects.trim(audio, top_db=35)

    if audio.size == 0:
        raise ValueError("That voice clip is too quiet. Please record closer to the microphone.")

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if peak <= 0.0 or rms < MIN_RMS:
        raise ValueError("That voice clip is too quiet. Please record in a quiet room, closer to the microphone.")

    duration = audio.size / REFERENCE_SAMPLE_RATE
    if duration < MIN_REFERENCE_SECONDS:
        raise ValueError("Please record at least 5 seconds of clear speech.")
    if duration > MAX_REFERENCE_SECONDS:
        max_samples = int(MAX_REFERENCE_SECONDS * REFERENCE_SAMPLE_RATE)
        audio = audio[:max_samples]

    audio = np.clip(audio, -1.0, 1.0)
    fd, cleaned_path = tempfile.mkstemp(prefix="dreamvoice_ref_", suffix=".wav")
    os.close(fd)
    sf.write(cleaned_path, audio, REFERENCE_SAMPLE_RATE, subtype="PCM_16")
    return cleaned_path
