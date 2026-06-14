"""English -> Kannada translation via AI4Bharat IndicTrans2 (Modal GPU).

Generating a story in English and translating to Kannada is far more reliable than
asking a small LLM to write Kannada directly. IndicTrans2 is SOTA for En->Indic.
"""

from __future__ import annotations


def translate_to_kannada(en_text: str) -> str:
    """Translate English story text to Kannada (Devanagari-free, Kannada script)."""
    text = (en_text or "").strip()
    if not text:
        raise ValueError("Nothing to translate.")

    import modal

    try:
        fn = modal.Function.from_name("dreamvoice-tts", "translate_en_kn")
        kn = fn.remote(text)
    except Exception as exc:  # noqa: BLE001 - surface a user-actionable app error.
        raise RuntimeError(
            "Kannada translation failed. Confirm `modal deploy modal_app.py` has run. "
            f"Original error: {exc}"
        ) from exc

    kn = (kn or "").strip()
    if not kn:
        raise RuntimeError("Translation returned empty Kannada text.")
    return kn
