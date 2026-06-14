"""Audio post-processing: silence trimming, normalization, quality checks."""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -35.0,
    min_silence_sec: float = 0.12,
    max_silence_sec: float = 0.20,
    keep_start: float = 0.03,
    keep_end: float = 0.05,
) -> np.ndarray:
    """Trim leading/trailing silence and compress long pauses between speech segments.

    Args:
        audio: 1-D float32 waveform.
        sample_rate: Samples per second.
        threshold_db: Silence threshold in dB (below this = silence).
        min_silence_sec: Minimum pause to keep (natural breathing room).
        max_silence_sec: Maximum pause allowed — anything longer gets capped here.
        keep_start: Seconds of silence to keep at the start.
        keep_end: Seconds of silence to keep at the end.

    Returns:
        Cleaned waveform (float32).
    """
    if audio.size == 0:
        return audio

    threshold = 10 ** (threshold_db / 20.0)
    sr = sample_rate

    # --- find voiced regions ---
    is_voiced = np.abs(audio) > threshold
    voiced_indices = np.where(is_voiced)[0]

    if voiced_indices.size == 0:
        return audio[: int(sr * 0.1)]

    # keep a small buffer before first voice and after last voice
    start = max(0, int(voiced_indices[0] - sr * keep_start))
    end = min(len(audio), int(voiced_indices[-1] + sr * keep_end))
    trimmed = audio[start:end].copy()

    # --- compress long inter-voice pauses ---
    target_pause = int(min_silence_sec * sr)
    max_pause = int(max_silence_sec * sr)
    result_parts: list[np.ndarray] = []
    fade_len = int(0.005 * sr)  # 5ms crossfade to avoid clicks
    i = 0
    n = len(trimmed)

    while i < n:
        if np.abs(trimmed[i]) <= threshold:
            silence_end = i
            while silence_end < n and np.abs(trimmed[silence_end]) <= threshold:
                silence_end += 1
            silence_len = silence_end - i
            if silence_len > max_pause:
                result_parts.append(trimmed[i: i + target_pause])
            elif silence_len > target_pause:
                result_parts.append(trimmed[i: i + target_pause])
            else:
                result_parts.append(trimmed[i: silence_end])
            i = silence_end
        else:
            voice_end = i
            while voice_end < n and np.abs(trimmed[voice_end]) > threshold:
                voice_end += 1
            result_parts.append(trimmed[i: voice_end])
            i = voice_end

    if not result_parts:
        return trimmed

    out = np.concatenate(result_parts).astype(np.float32)

    # fade-in / fade-out to avoid clicks at trim points
    if len(out) > fade_len * 2:
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        out[:fade_len] *= fade_in
        out[-fade_len:] *= fade_out

    return out


def normalize_audio(
    audio: np.ndarray,
    target_peak_db: float = -1.0,
) -> np.ndarray:
    """Peak-normalize audio to a target level."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    target = 10 ** (target_peak_db / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def postprocess(
    audio: np.ndarray,
    sample_rate: int,
    trim: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Full post-processing pipeline: trim silence + normalize."""
    if audio.size == 0:
        return audio
    if trim:
        audio = trim_silence(audio, sample_rate)
    if normalize:
        audio = normalize_audio(audio)
    return audio


def postprocess_wav_bytes(
    wav_bytes: bytes,
    sample_rate: int,
    trim: bool = True,
    normalize: bool = True,
) -> bytes:
    """Post-process WAV bytes (used on Modal GPU before sending back)."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    audio = postprocess(audio, sr, trim=trim, normalize=normalize)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()
