"""VoxCPM2 voice cloning and narration — local GPU inference for HF Spaces."""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np

MODEL_ID = "openbmb/VoxCPM2"

_model = None


def _get_model():
    global _model
    if _model is None:
        from voxcpm import VoxCPM
        _model = VoxCPM.from_pretrained(
            MODEL_ID,
            device="cuda",
            load_denoiser=True,
            optimize=True,
        )
    return _model


def _split_sentences(text: str, max_chars: int = 200):
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        out.append(p)
    return out or [text.strip()]


def _pause_for(mood: str, energy: float = 0.45) -> float:
    energy = max(0.0, min(1.0, float(energy)))
    base = 0.45 if mood in ("funny", "magical") else 0.65
    return round(base + (0.85 - base) * (1.0 - energy), 3)


def _with_bedtime_style(text: str, speed: float, mood: str = "", energy: float = 0.45) -> str:
    energy = max(0.0, min(1.0, float(energy)))
    mood_styles = {
        "magical": "gentle, warm, slightly slow, wonder-filled whisper, like telling a secret about something beautiful, soft rising intonation on wonder words, pause briefly after each beat",
        "funny": "warm, playful, slightly animated, gentle humor in the voice, light chuckle between lines, bright and cheerful but still soft enough for bedtime, slightly faster pace than usual",
        "calming": "very slow, deep warm whisper, barely above a breath, each word drifting gently into the next, long pauses between sentences, voice fading softly at the end of each line, like someone falling asleep while reading",
        "dreamy": "slow, soft, breathy whisper, voice drifting like floating on a cloud, elongated vowels, gentle hum between phrases, lullaby-like rhythm, words dissolving into silence",
    }
    style = mood_styles.get(mood, "gentle, warm, sleepy bedtime voice, slightly slow pace")
    if energy >= 0.66:
        style += ", a little brighter and more animated, lively for a delighted child"
    elif energy <= 0.33:
        style += ", even softer and slower, barely above a whisper"
    return f"({style}){text}"


def _postprocess_np(audio, sr):
    from audio_postprocess import postprocess
    return postprocess(audio, sr)


def clone_and_speak(ref_wav: str, text: str, speed: float = 0.9, mood: str = "", energy: float = 0.45) -> str:
    """Clone the reference voice and synthesize text to a temporary WAV path.

    Runs on local GPU (HF Spaces GPU Zero).
    """
    if not ref_wav or not os.path.exists(ref_wav):
        raise ValueError("Please provide a prepared voice reference WAV.")

    story_text = (text or "").strip()
    if not story_text:
        raise ValueError("Please provide story text to narrate.")

    model = _get_model()
    sr = int(model.tts_model.sample_rate)

    pause = _pause_for(mood, energy)
    silence = np.zeros(int(pause * sr), dtype=np.float32)

    chunks = []
    for sentence in _split_sentences(story_text):
        wav = model.generate(
            text=_with_bedtime_style(sentence, speed, mood, energy),
            reference_wav_path=ref_wav,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=True,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=8.0,
        )
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size:
            chunks.append(wav)
            chunks.append(silence)

    if not chunks:
        raise RuntimeError("VoxCPM2 produced no audio.")

    full = np.concatenate(chunks)
    full = _postprocess_np(full, sr)

    import soundfile as sf
    fd, out_path = tempfile.mkstemp(prefix="dreamvoice_story_", suffix=".wav")
    os.close(fd)
    sf.write(out_path, full, sr)
    return out_path
