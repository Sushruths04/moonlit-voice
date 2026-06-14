"""Modal GPU functions for DreamVoice VoxCPM2 synthesis."""

from __future__ import annotations

import io
import os
import tempfile

import modal
import numpy as np

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
def synthesize_story(
    ref_wav_bytes: bytes, text: str, speed: float = 0.9, mood: str = "", energy: float = 0.45
) -> bytes:
    """Synthesize story narration on a Modal GPU and return WAV bytes.

    *mood* adds expressive style tags (e.g. magical → whimsical whisper,
    funny → playful animated). *energy* (0=calm..1=lively) shapes delivery + pauses.
    """
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

        import numpy as np

        model = _load_model()
        sr = int(model.tts_model.sample_rate)
        # Sentence-by-sentence synthesis with gentle pauses → natural, lively prosody
        # instead of one flat, continuous, monotone pass. Higher energy → shorter pauses.
        pause = _pause_for(mood, energy)
        silence = np.zeros(int(pause * sr), dtype=np.float32)

        chunks = []
        for sentence in _split_sentences(story_text):
            wav = model.generate(
                text=_with_bedtime_style(sentence, speed, mood, energy),
                reference_wav_path=ref_path,
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
        return _wav_to_bytes(full, sr)
    finally:
        try:
            os.remove(ref_path)
        except FileNotFoundError:
            pass


def _pause_for(mood: str, energy: float = 0.45) -> float:
    """Inter-sentence pause (seconds): shorter when energetic, longer when calm."""
    energy = max(0.0, min(1.0, float(energy)))
    base = 0.22 if mood in ("funny", "magical") else 0.34
    return round(base + (0.42 - base) * (1.0 - energy), 3)  # ~0.22..0.42s


def _with_bedtime_style(text: str, speed: float, mood: str = "", energy: float = 0.45) -> str:
    """Wrap text with VoxCPM2 parenthetical style tags for expressive narration.

    Style tags control emotion, pace, and expression. VoxCPM2 reads the parenthetical
    description and adjusts its delivery accordingly. *energy* nudges it brighter/softer.
    """
    energy = max(0.0, min(1.0, float(energy)))
    mood_styles = {
        "magical": (
            "gentle, warm, slightly slow, wonder-filled whisper, "
            "like telling a secret about something beautiful, "
            "soft rising intonation on wonder words, "
            "pause briefly after each beat"
        ),
        "funny": (
            "warm, playful, slightly animated, "
            "gentle humor in the voice, light chuckle between lines, "
            "bright and cheerful but still soft enough for bedtime, "
            "slightly faster pace than usual"
        ),
        "calming": (
            "very slow, deep warm whisper, "
            "barely above a breath, each word drifting gently into the next, "
            "long pauses between sentences, "
            "voice fading softly at the end of each line, "
            "like someone falling asleep while reading"
        ),
        "dreamy": (
            "slow, soft, breathy whisper, "
            "voice drifting like floating on a cloud, "
            "elongated vowels, gentle hum between phrases, "
            "lullaby-like rhythm, words dissolving into silence"
        ),
    }
    style = mood_styles.get(mood, "gentle, warm, sleepy bedtime voice, slightly slow pace")
    if energy >= 0.66:
        style += ", a little brighter and more animated, lively for a delighted child"
    elif energy <= 0.33:
        style += ", even softer and slower, barely above a whisper"
    return f"({style}){text}"


def _wav_to_bytes(wav, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()


def _postprocess_np(audio, sr):
    """Trim silence + peak-normalize on Modal GPU before sending back."""
    import numpy as _np
    audio = _np.asarray(audio, dtype=_np.float32)
    threshold_db = -35.0
    min_silence_sec = 0.12
    max_silence_sec = 0.20
    keep_start = 0.03
    keep_end = 0.05
    target_peak_db = -1.0

    if audio.size == 0:
        return audio

    threshold = 10 ** (threshold_db / 20.0)
    is_voiced = _np.abs(audio) > threshold
    voiced_indices = _np.where(is_voiced)[0]

    if voiced_indices.size == 0:
        return audio[: int(sr * 0.1)]

    start = max(0, int(voiced_indices[0] - sr * keep_start))
    end = min(len(audio), int(voiced_indices[-1] + sr * keep_end))
    trimmed = audio[start:end].copy()

    target_pause = int(min_silence_sec * sr)
    max_pause = int(max_silence_sec * sr)
    result_parts = []
    fade_len = int(0.005 * sr)
    i = 0
    n = len(trimmed)
    while i < n:
        if _np.abs(trimmed[i]) <= threshold:
            silence_end = i
            while silence_end < n and _np.abs(trimmed[silence_end]) <= threshold:
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
            while voice_end < n and _np.abs(trimmed[voice_end]) > threshold:
                voice_end += 1
            result_parts.append(trimmed[i: voice_end])
            i = voice_end

    if result_parts:
        trimmed = _np.concatenate(result_parts).astype(_np.float32)

    if len(trimmed) > fade_len * 2:
        fade_in = _np.linspace(0, 1, fade_len, dtype=_np.float32)
        fade_out = _np.linspace(1, 0, fade_len, dtype=_np.float32)
        trimmed[:fade_len] *= fade_in
        trimmed[-fade_len:] *= fade_out

    peak = float(_np.max(_np.abs(trimmed)))
    if peak > 1e-6:
        target = 10 ** (target_peak_db / 20.0)
        trimmed = (trimmed * (target / peak)).astype(_np.float32)

    return trimmed


# ════════════════════════════════════════════════════════════════════════════
# Kannada pipeline — IndicF5 narration + IndicTrans2 translation (AI4Bharat)
# ════════════════════════════════════════════════════════════════════════════

INDICF5_ID = "ai4bharat/IndicF5"
INDICF5_SR = 24_000
INDICTRANS_ID = "ai4bharat/indictrans2-en-indic-1B"

indicf5_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers==4.46.3",
        "soundfile==0.14.0",
        "numpy==1.26.4",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

indictrans_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0",
        "transformers==4.51.3",
        "sentencepiece==0.2.0",
    )
    .pip_install("git+https://github.com/VarunGumma/IndicTransToolkit.git")
)

_indicf5 = None
_indictrans = None


def _load_indicf5():
    global _indicf5
    if _indicf5 is None:
        from transformers import AutoModel

        token = os.environ.get("HF_TOKEN") or None
        _indicf5 = AutoModel.from_pretrained(
            INDICF5_ID, trust_remote_code=True, cache_dir="/cache", token=token
        )
        try:
            _indicf5 = _indicf5.to("cuda")
        except Exception:
            pass
    return _indicf5


def _load_indictrans():
    global _indictrans
    if _indictrans is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN") or None
        tok = AutoTokenizer.from_pretrained(INDICTRANS_ID, trust_remote_code=True, cache_dir="/cache", token=token)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS_ID, trust_remote_code=True, cache_dir="/cache", token=token
        )
        try:
            model = model.to("cuda")
        except Exception:
            pass
        _indictrans = (tok, model)
    return _indictrans


def _split_sentences(text: str, max_chars: int = 240):
    """Split into speakable units on English + Kannada (danda ।) terminators."""
    import re

    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Hard-wrap over-long sentences so TTS prosody stays natural.
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        out.append(p)
    return out or [text.strip()]


@app.function(image=indictrans_image, gpu="A10G", timeout=600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("dreamvoice-secrets")])
def translate_en_kn(text: str) -> str:
    """Translate English story text to Kannada with IndicTrans2."""
    import torch
    from IndicTransToolkit.processor import IndicProcessor

    text = (text or "").strip()
    if not text:
        raise ValueError("Text to translate is required.")

    tok, model = _load_indictrans()
    ip = IndicProcessor(inference=True)
    sents = _split_sentences(text)

    batch = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_length=256, num_beams=5, num_return_sequences=1)
    decoded = tok.batch_decode(generated, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang="kan_Knda")
    return " ".join(t.strip() for t in translations if t.strip())


@app.function(image=indicf5_image, gpu="A10G", timeout=900, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("dreamvoice-secrets")])
def synthesize_kannada(
    ref_wav_bytes: bytes, ref_text: str, kannada_text: str, mood: str = "", energy: float = 0.45
) -> bytes:
    """Narrate Kannada text in the cloned voice, sentence-by-sentence for natural prosody."""
    import numpy as np

    if not ref_wav_bytes:
        raise ValueError("Reference WAV bytes are required.")
    if not (ref_text or "").strip():
        raise ValueError("Reference transcript is required for IndicF5.")
    if not (kannada_text or "").strip():
        raise ValueError("Kannada text is required.")

    fd, ref_path = tempfile.mkstemp(prefix="indicf5_ref_", suffix=".wav")
    os.close(fd)
    try:
        with open(ref_path, "wb") as fh:
            fh.write(ref_wav_bytes)

        model = _load_indicf5()
        silence = np.zeros(int(_pause_for(mood, energy) * INDICF5_SR), dtype=np.float32)

        chunks = []
        for sentence in _split_sentences(kannada_text):
            audio = model(sentence, ref_audio_path=ref_path, ref_text=ref_text.strip())
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size and float(np.max(np.abs(audio))) > 1.0:  # int16-range -> normalize
                audio = audio / 32768.0
            if audio.size:
                chunks.append(audio)
                chunks.append(silence)

        if not chunks:
            raise RuntimeError("IndicF5 produced no audio.")
        full = np.concatenate(chunks)
        full = _postprocess_np(full, INDICF5_SR)
        return _wav_to_bytes(full, INDICF5_SR)
    finally:
        try:
            os.remove(ref_path)
        except FileNotFoundError:
            pass
