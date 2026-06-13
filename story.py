"""Kid-safe bedtime story generation with MiniCPM5-1B."""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openbmb/MiniCPM5-1B"
SYSTEM_PROMPT = (
    "Write a bedtime story for a 6-year-old. <=150 words. Structure: a gentle hook, "
    "three short story beats, a calming resolution, and a final sleepy good-night line. "
    "Warm and simple. No violence, no fear, no death, nothing scary."
)
VALID_GENRES = {"animals", "kingdom", "space", "dragons", "ocean", "forest"}
VALID_MOODS = {"magical", "funny", "calming", "dreamy"}

_tokenizer: Any | None = None
_model: Any | None = None


def _get_model() -> tuple[Any, Any]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        token = os.environ.get("HF_TOKEN") or None
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            token=token,
        )
        _model.eval()
    return _tokenizer, _model


def generate_story(genre: str, mood: str, hero_name: str = "", language: str = "en") -> str:
    """Generate a short, calm bedtime story."""
    genre = _clean_choice(genre, VALID_GENRES, "genre")
    mood = _clean_choice(mood, VALID_MOODS, "mood")
    hero_name = hero_name.strip()[:40]
    language = (language or "en").strip()

    user_prompt = (
        f"Genre: {genre}\n"
        f"Mood: {mood}\n"
        f"Hero name: {hero_name or 'a kind child'}\n"
        f"Language: {language}\n"
        "Return only the story text. Keep it gentle, concrete, and ready for voice narration."
    )

    tokenizer, model = _get_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=230,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    story = _normalize_story(text)
    if not story:
        raise RuntimeError("MiniCPM5 returned an empty story. Please try again.")
    return story


def _clean_choice(value: str, allowed: set[str], label: str) -> str:
    cleaned = (value or "").strip().lower()
    if cleaned not in allowed:
        raise ValueError(f"Please choose a supported {label}: {', '.join(sorted(allowed))}.")
    return cleaned


def _normalize_story(text: str) -> str:
    story = re.sub(r"\s+", " ", text).strip().strip('"')
    if story.lower().startswith("story:"):
        story = story[6:].strip()

    words = story.split()
    if len(words) <= 165:
        return story

    trimmed = " ".join(words[:165])
    last_sentence_end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if last_sentence_end > 80:
        return trimmed[: last_sentence_end + 1].strip()
    return trimmed.rstrip(" ,;:") + "."
