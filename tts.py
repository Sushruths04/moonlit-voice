"""VoxCPM2 voice cloning and narration."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

import soundfile as sf
import torch
from voxcpm import VoxCPM

MODEL_ID = "openbmb/VoxCPM2"

_model: Any | None = None


def _get_model() -> Any:
    global _model
    if _model is None:
        token = os.environ.get("HF_TOKEN") or None
        device = "cuda" if _cuda_is_usable() else "cpu"
        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        _model = VoxCPM.from_pretrained(
            MODEL_ID,
            device=device,
            load_denoiser=device == "cuda",
            optimize=device == "cuda",
            **kwargs,
        )
    return _model


def clone_and_speak(ref_wav: str, text: str, speed: float = 0.9) -> str:
    """Clone the reference voice and synthesize text to a temporary WAV path."""
    if not ref_wav or not os.path.exists(ref_wav):
        raise ValueError("Please provide a prepared voice reference WAV.")

    story_text = (text or "").strip()
    if not story_text:
        raise ValueError("Please provide story text to narrate.")

    backend = os.environ.get("DREAMVOICE_TTS_BACKEND", "modal").strip().lower()
    if backend == "modal":
        return _clone_and_speak_modal(ref_wav, story_text, speed)
    if backend == "local":
        return _clone_and_speak_local(ref_wav, story_text, speed)
    raise ValueError("DREAMVOICE_TTS_BACKEND must be either 'modal' or 'local'.")


def _clone_and_speak_modal(ref_wav: str, text: str, speed: float) -> str:
    import modal

    with open(ref_wav, "rb") as ref_file:
        ref_wav_bytes = ref_file.read()

    try:
        synthesize_story = modal.Function.from_name("dreamvoice-tts", "synthesize_story")
        wav_bytes = synthesize_story.remote(ref_wav_bytes, text, speed)
    except Exception as exc:  # noqa: BLE001 - surface a user-actionable app error.
        raise RuntimeError(
            "Modal VoxCPM2 synthesis failed. Confirm `modal deploy modal_app.py` has run "
            f"and Modal credentials are configured. Original error: {exc}"
        ) from exc

    fd, out_path = tempfile.mkstemp(prefix="dreamvoice_story_", suffix=".wav")
    os.close(fd)
    with open(out_path, "wb") as out_file:
        out_file.write(wav_bytes)
    return out_path


def _clone_and_speak_local(ref_wav: str, text: str, speed: float) -> str:
    bedtime_text = _with_bedtime_style(text, speed)
    model = _get_model()
    wav = model.generate(
        text=bedtime_text,
        reference_wav_path=ref_wav,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=True,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=8.0,
    )

    sample_rate = int(getattr(model.tts_model, "sample_rate", 48_000))
    fd, out_path = tempfile.mkstemp(prefix="dreamvoice_story_", suffix=".wav")
    os.close(fd)
    sf.write(out_path, wav, sample_rate)
    return out_path


def _cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        torch.empty(1, device="cuda")
    except Exception:
        return False
    return True


def _with_bedtime_style(text: str, speed: float) -> str:
    style = "gentle, warm, sleepy bedtime voice"
    if speed < 0.95:
        style += ", slightly slow pace"
    elif speed > 1.05:
        style += ", slightly lively pace"
    return f"({style}){text}"
