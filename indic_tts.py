"""Kannada narration via AI4Bharat IndicF5 — local GPU for HF Spaces.

Uses the fine-tuned checkpoint from mitvho09/IndicF5-Kannada-Bedtime-v2
(best Kannada quality: MOS 4.2, speaking rate 3.0 syll/s).
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import torch

INDICF5_ID = "ai4bharat/IndicF5"
INDICF5_V2_REPO = "mitvho09/IndicF5-Kannada-Bedtime-v2"
INDICF5_SR = 24_000

_model = None


def _get_model():
    global _model
    if _model is None:
        from transformers import AutoModel
        token = os.environ.get("HF_TOKEN") or None

        _model = AutoModel.from_pretrained(
            INDICF5_ID, trust_remote_code=True, token=token)

        # Load fine-tuned Kannada checkpoint from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            cfm_path = hf_hub_download(
                repo_id=INDICF5_V2_REPO,
                filename="cfm.pt",
                token=token,
            )
            cfm_state = torch.load(cfm_path, map_location="cpu", weights_only=True)
            _model.ema_model.load_state_dict(cfm_state)
            print(f"✓ Loaded fine-tuned CFM from {INDICF5_V2_REPO}")
        except Exception as e:
            print(f"⚠ Could not load fine-tuned checkpoint: {e}")

        _model = _model.to("cuda")
        _model.eval()
    return _model


# Load at module level for ZeroGPU (CUDA emulation outside @spaces.GPU)
try:
    _get_model()
except Exception:
    pass


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


def _postprocess_np(audio, sr):
    from audio_postprocess import postprocess
    return postprocess(audio, sr)


def narrate_kannada(ref_wav: str, ref_text: str, kannada_text: str, mood: str = "", energy: float = 0.45) -> str:
    """Clone the parent's voice and narrate Kannada text. Returns a temp WAV path."""
    if not ref_wav or not os.path.exists(ref_wav):
        raise ValueError("Please provide a prepared voice reference WAV.")
    if not (ref_text or "").strip():
        raise ValueError("Reference transcript (ref_text) is required for Kannada cloning.")
    if not (kannada_text or "").strip():
        raise ValueError("Please provide Kannada text to narrate.")

    model = _get_model()

    pause = _pause_for(mood, energy) * 1.05
    silence = np.zeros(int(pause * INDICF5_SR), dtype=np.float32)

    chunks = []
    for sentence in _split_sentences(kannada_text, max_chars=200):
        audio = model(sentence, ref_audio_path=ref_wav, ref_text=ref_text.strip())
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size and float(np.max(np.abs(audio))) > 1.0:
            audio = audio / 32768.0
        if audio.size:
            chunks.append(audio)
            chunks.append(silence)

    if not chunks:
        raise RuntimeError("IndicF5 produced no audio.")

    full = np.concatenate(chunks)
    full = _postprocess_np(full, INDICF5_SR)

    import soundfile as sf
    fd, out_path = tempfile.mkstemp(prefix="dreamvoice_kn_", suffix=".wav")
    os.close(fd)
    sf.write(out_path, full, INDICF5_SR)
    return out_path
