"""English -> Kannada translation via AI4Bharat IndicTrans2 — local GPU for HF Spaces."""

from __future__ import annotations

import os
import re

import torch

INDICTRANS_ID = "ai4bharat/indictrans2-en-indic-1B"

_tok = None
_model = None


def _get_model():
    global _tok, _model
    if _tok is None or _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        token = os.environ.get("HF_TOKEN") or None
        _tok = AutoTokenizer.from_pretrained(
            INDICTRANS_ID, trust_remote_code=True, token=token)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS_ID, trust_remote_code=True, token=token)
        _model = _model.to("cuda")
        _model.eval()
    return _tok, _model


# Load at module level for ZeroGPU (CUDA emulation outside @spaces.GPU)
try:
    _get_model()
except Exception:
    pass


def _split_sentences(text: str, max_chars: int = 180):
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


def translate_to_kannada(en_text: str) -> str:
    """Translate English story text to Kannada (Kannada script)."""
    text = (en_text or "").strip()
    if not text:
        raise ValueError("Nothing to translate.")

    from IndicTransToolkit.processor import IndicProcessor

    tok, model = _get_model()
    ip = IndicProcessor(inference=True)

    sents = _split_sentences(text, max_chars=180)
    batch = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_length=512, num_beams=5,
            num_return_sequences=1, length_penalty=1.0,
        )
    decoded = tok.batch_decode(generated, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang="kan_Knda")

    kn = " ".join(t.strip() for t in translations if t.strip())
    if not kn:
        raise RuntimeError("Translation returned empty Kannada text.")
    return kn
