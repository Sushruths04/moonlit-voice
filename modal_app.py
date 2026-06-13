"""Modal GPU functions for DreamVoice VoxCPM2 synthesis."""

from __future__ import annotations

import io
import os
import tempfile

import modal

MODEL_ID = "openbmb/VoxCPM2"
APP_NAME = "dreamvoice-tts"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "voxcpm==2.0.3",
        "soundfile==0.14.0",
        "numpy==2.2.6",
        "huggingface_hub==1.19.0",
    )
)

hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)

_model = None


def _load_model():
    global _model
    if _model is None:
        from voxcpm import VoxCPM

        _model = VoxCPM.from_pretrained(
            MODEL_ID,
            cache_dir="/cache",
            device="cuda",
            load_denoiser=True,
            optimize=True,
        )
    return _model


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
    volumes={"/cache": hf_cache},
)
def synthesize_reference(text: str = "This is a gentle reference voice for DreamVoice testing.") -> bytes:
    """Create a disposable speech reference clip for smoke tests."""
    story_text = (text or "").strip()
    if not story_text:
        raise ValueError("Reference text is required.")

    model = _load_model()
    wav = model.generate(
        text=f"(A warm adult bedtime narrator voice){story_text}",
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=True,
        retry_badcase=True,
        retry_badcase_max_times=3,
    )
    return _wav_to_bytes(wav, int(model.tts_model.sample_rate))


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
    volumes={"/cache": hf_cache},
)
def synthesize_story(ref_wav_bytes: bytes, text: str, speed: float = 0.9) -> bytes:
    """Synthesize story narration on a Modal GPU and return WAV bytes."""
    if not ref_wav_bytes:
        raise ValueError("Reference WAV bytes are required for voice cloning.")
    story_text = (text or "").strip()
    if not story_text:
        raise ValueError("Story text is required for narration.")

    fd, ref_path = tempfile.mkstemp(prefix="dreamvoice_modal_ref_", suffix=".wav")
    os.close(fd)
    try:
        with open(ref_path, "wb") as ref_file:
            ref_file.write(ref_wav_bytes)

        model = _load_model()
        wav = model.generate(
            text=_with_bedtime_style(story_text, speed),
            reference_wav_path=ref_path,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=True,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=8.0,
        )

        return _wav_to_bytes(wav, int(model.tts_model.sample_rate))
    finally:
        try:
            os.remove(ref_path)
        except FileNotFoundError:
            pass


def _with_bedtime_style(text: str, speed: float) -> str:
    style = "gentle, warm, sleepy bedtime voice"
    if speed < 0.95:
        style += ", slightly slow pace"
    elif speed > 1.05:
        style += ", slightly lively pace"
    return f"({style}){text}"


def _wav_to_bytes(wav, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()
