"""Modal GPU functions for DreamVoice — warm containers for speed.

All three pipelines (English VoxCPM2, Kannada IndicF5, IndicTrans2) use
@app.cls with keep_warm=1 so models stay loaded between requests.
First call: ~2-3 min cold start. After that: seconds.
"""

from __future__ import annotations

import io
import os
import re
import tempfile

import modal
import numpy as np

APP_NAME = "dreamvoice-tts"
hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

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


def _wav_to_bytes(wav, sample_rate: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


def _postprocess_np(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    threshold = 10 ** (-35.0 / 20.0)
    is_voiced = np.abs(audio) > threshold
    voiced = np.where(is_voiced)[0]
    if voiced.size == 0:
        return audio[: int(sr * 0.1)]
    start = max(0, int(voiced[0] - sr * 0.03))
    end = min(len(audio), int(voiced[-1] + sr * 0.05))
    trimmed = audio[start:end].copy()
    max_pause = int(0.20 * sr)
    target_pause = int(0.12 * sr)
    parts = []
    i, n = 0, len(trimmed)
    while i < n:
        if np.abs(trimmed[i]) <= threshold:
            j = i
            while j < n and np.abs(trimmed[j]) <= threshold:
                j += 1
            parts.append(trimmed[i: min(i + max_pause, j)])
            i = j
        else:
            j = i
            while j < n and np.abs(trimmed[j]) > threshold:
                j += 1
            parts.append(trimmed[i:j])
            i = j
    if parts:
        trimmed = np.concatenate(parts).astype(np.float32)
    fade = int(0.005 * sr)
    if len(trimmed) > fade * 2:
        trimmed[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        trimmed[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = float(np.max(np.abs(trimmed)))
    if peak > 1e-6:
        target = 10 ** (-1.0 / 20.0)
        trimmed = (trimmed * (target / peak)).astype(np.float32)
    return trimmed


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


# ════════════════════════════════════════════════════════════════════════════
# VoxCPM2 — English TTS (warm container)
# ════════════════════════════════════════════════════════════════════════════

vox_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.11.0", "torchaudio==2.11.0", "voxcpm==2.0.3",
        "soundfile==0.14.0", "numpy==2.2.6", "huggingface_hub==1.19.0",
    )
)


@app.cls(image=vox_image, gpu="A10G", timeout=900,
         volumes={"/cache": hf_cache}, min_containers=1)
class VoxTTS:
    @modal.enter()
    def load(self):
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2", cache_dir="/cache",
            device="cuda", load_denoiser=True, optimize=True,
        )
        self.sr = int(self.model.tts_model.sample_rate)

    @modal.method()
    def synthesize(self, ref_wav_bytes: bytes, text: str, speed: float = 0.9,
                   mood: str = "", energy: float = 0.45) -> bytes:
        if not ref_wav_bytes:
            raise ValueError("Reference WAV bytes are required.")
        story_text = (text or "").strip()
        if not story_text:
            raise ValueError("Story text is required.")

        fd, ref_path = tempfile.mkstemp(prefix="vox_ref_", suffix=".wav")
        os.close(fd)
        try:
            with open(ref_path, "wb") as f:
                f.write(ref_wav_bytes)

            pause = _pause_for(mood, energy)
            silence = np.zeros(int(pause * self.sr), dtype=np.float32)

            chunks = []
            for sentence in _split_sentences(story_text):
                wav = self.model.generate(
                    text=_with_bedtime_style(sentence, speed, mood, energy),
                    reference_wav_path=ref_path,
                    cfg_value=2.0, inference_timesteps=10,
                    normalize=True, denoise=True,
                    retry_badcase=True, retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=8.0,
                )
                wav = np.asarray(wav, dtype=np.float32)
                if wav.size:
                    chunks.append(wav)
                    chunks.append(silence)

            if not chunks:
                raise RuntimeError("VoxCPM2 produced no audio.")
            full = np.concatenate(chunks)
            full = _postprocess_np(full, self.sr)
            return _wav_to_bytes(full, self.sr)
        finally:
            try:
                os.remove(ref_path)
            except FileNotFoundError:
                pass


# ════════════════════════════════════════════════════════════════════════════
# IndicTrans2 — English → Kannada translation (warm container)
# ════════════════════════════════════════════════════════════════════════════

INDICTRANS_ID = "ai4bharat/indictrans2-en-indic-1B"

trans_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0", "transformers==4.51.3", "sentencepiece==0.2.0",
    )
    .pip_install("git+https://github.com/VarunGumma/IndicTransToolkit.git")
)


@app.cls(image=trans_image, gpu="A10G", timeout=600,
         volumes={"/cache": hf_cache}, min_containers=1,
         secrets=[modal.Secret.from_name("dreamvoice-secrets")])
class IndicTrans:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        token = os.environ.get("HF_TOKEN") or None
        self.tok = AutoTokenizer.from_pretrained(
            INDICTRANS_ID, trust_remote_code=True, cache_dir="/cache", token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS_ID, trust_remote_code=True, cache_dir="/cache", token=token)
        try:
            self.model = self.model.to("cuda")
        except Exception:
            pass

    @modal.method()
    def translate(self, text: str) -> str:
        from IndicTransToolkit.processor import IndicProcessor
        text = (text or "").strip()
        if not text:
            raise ValueError("Text to translate is required.")

        ip = IndicProcessor(inference=True)
        sents = _split_sentences(text, max_chars=180)
        batch = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="kan_Knda")
        inputs = self.tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.model.device)
        with __import__("torch").inference_mode():
            generated = self.model.generate(
                **inputs, max_length=512, num_beams=5,
                num_return_sequences=1, length_penalty=1.0,
            )
        decoded = self.tok.batch_decode(generated, skip_special_tokens=True)
        translations = ip.postprocess_batch(decoded, lang="kan_Knda")
        return " ".join(t.strip() for t in translations if t.strip())


# ════════════════════════════════════════════════════════════════════════════
# IndicF5 — Kannada TTS (warm container)
# ════════════════════════════════════════════════════════════════════════════

INDICF5_ID = "ai4bharat/IndicF5"
INDICF5_V2_REPO = "mitvho09/IndicF5-Kannada-Bedtime-v2"
INDICF5_SR = 24_000

f5_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4",
        "huggingface_hub",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)


@app.cls(image=f5_image, gpu="A10G", timeout=900,
         volumes={"/cache": hf_cache}, min_containers=1,
         secrets=[modal.Secret.from_name("dreamvoice-secrets")])
class IndicF5TTS:
    @modal.enter()
    def load(self):
        from transformers import AutoModel
        import torch
        token = os.environ.get("HF_TOKEN") or None
        self.model = AutoModel.from_pretrained(
            INDICF5_ID, trust_remote_code=True, cache_dir="/cache", token=token)
        # Load v2 fine-tuned Kannada checkpoint
        try:
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=INDICF5_V2_REPO, filename="model.safetensors",
                cache_dir="/cache", token=token)
            state = load_file(ckpt_path, device="cpu")
            # The v2 checkpoint has full model weights saved via save_pretrained
            # Keys like: ema_model._orig_mod.transformer.text_embed...
            # We need to load into the full model, not just ema_model
            self.model.load_state_dict(state, strict=False)
            print(f"✓ Loaded IndicF5 v2 fine-tuned weights ({len(state)} params)")
        except Exception as e:
            print(f"⚠ Could not load v2 checkpoint: {e} — using stock IndicF5")
        try:
            self.model = self.model.to("cuda")
        except Exception:
            pass

    @modal.method()
    def synthesize(self, ref_wav_bytes: bytes, ref_text: str,
                   kannada_text: str, mood: str = "", energy: float = 0.45) -> bytes:
        if not ref_wav_bytes:
            raise ValueError("Reference WAV bytes are required.")
        if not (ref_text or "").strip():
            raise ValueError("Reference transcript is required.")
        if not (kannada_text or "").strip():
            raise ValueError("Kannada text is required.")

        fd, ref_path = tempfile.mkstemp(prefix="f5_ref_", suffix=".wav")
        os.close(fd)
        try:
            with open(ref_path, "wb") as fh:
                fh.write(ref_wav_bytes)

            pause = _pause_for(mood, energy) * 1.3
            silence = np.zeros(int(pause * INDICF5_SR), dtype=np.float32)

            chunks = []
            for sentence in _split_sentences(kannada_text, max_chars=200):
                audio = self.model(sentence, ref_audio_path=ref_path, ref_text=ref_text.strip())
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
            return _wav_to_bytes(full, INDICF5_SR)
        finally:
            try:
                os.remove(ref_path)
            except FileNotFoundError:
                pass


# ════════════════════════════════════════════════════════════════════════════
# Compat shim — old function-based callers still work
# ════════════════════════════════════════════════════════════════════════════

_vox_cls = None
_trans_cls = None
_f5_cls = None


def _get_vox():
    global _vox_cls
    if _vox_cls is None:
        _vox_cls = VoxTTS()
    return _vox_cls


def _get_trans():
    global _trans_cls
    if _trans_cls is None:
        _trans_cls = IndicTrans()
    return _trans_cls


def _get_f5():
    global _f5_cls
    if _f5_cls is None:
        _f5_cls = IndicF5TTS()
    return _f5_cls


@app.function(image=vox_image, gpu="A10G", timeout=900, volumes={"/cache": hf_cache})
def synthesize_reference(text: str = "This is a gentle reference voice for DreamVoice testing.") -> bytes:
    story_text = (text or "").strip()
    if not story_text:
        raise ValueError("Reference text is required.")
    vox = _get_vox()
    return vox.synthesize.remote(
        b"\x00" * 100,  # dummy ref
        f"(A warm adult bedtime narrator voice){story_text}",
    )


@app.function(image=vox_image, gpu="A10G", timeout=900, volumes={"/cache": hf_cache})
def synthesize_story(
    ref_wav_bytes: bytes, text: str, speed: float = 0.9, mood: str = "", energy: float = 0.45
) -> bytes:
    vox = _get_vox()
    return vox.synthesize.remote(ref_wav_bytes, text, speed, mood, energy)


@app.function(image=trans_image, gpu="A10G", timeout=600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("dreamvoice-secrets")])
def translate_en_kn(text: str) -> str:
    trans = _get_trans()
    return trans.translate.remote(text)


@app.function(image=f5_image, gpu="A10G", timeout=900, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("dreamvoice-secrets")])
def synthesize_kannada(
    ref_wav_bytes: bytes, ref_text: str, kannada_text: str, mood: str = "", energy: float = 0.45
) -> bytes:
    f5 = _get_f5()
    return f5.synthesize.remote(ref_wav_bytes, ref_text, kannada_text, mood, energy)
