"""Kid-safe bedtime story generation with MiniCPM (length-reliable).

Generates in ENGLISH only. Kannada/other languages are produced by translating
this English story downstream (see indic_text.translate_to_kannada), which is far
more reliable than asking a small model to write directly in a low-resource language.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# MiniCPM5-1B is the low-RAM default. Set DREAMVOICE_STORY_MODEL=openbmb/MiniCPM4.1-8B
# (still OpenBMB, <=32B) for noticeably richer, longer stories — run it on a GPU.
MODEL_ID = os.environ.get("DREAMVOICE_STORY_MODEL", "openbmb/MiniCPM5-1B")

SYSTEM_PROMPT = (
    "You are a warm, lively bedtime storyteller for young children, like a parent reading aloud "
    "with gentle energy. Tell vivid, playful stories full of friendly characters, little sounds "
    "(whoosh, splash, giggle), and small moments of wonder — then slow down and soften toward a "
    "calm, sleepy ending.\n"
    "STRICT RULES:\n"
    "- Plain text only. No markdown, no asterisks, no bullet points, no emoji, no headings.\n"
    "- Real, meaningful sentences with concrete detail — never filler or repetition.\n"
    "- Structure: a gentle hook, three story beats with a little dialogue, a calming resolution, "
    "and a final drowsy good-night line.\n"
    "- Use '...' for natural pauses between beats.\n"
    "- Kid-safe always: no violence, no fear, no death, nothing scary.\n"
    "- Do NOT restate these instructions or write a title — output only the story."
)

VALID_GENRES = {"animals", "kingdom", "space", "dragons", "ocean", "forest"}
VALID_MOODS = {"magical", "funny", "calming", "dreamy"}

LENGTH_PRESETS = {
    "1 min": {"target_words": 150, "max_new_tokens": 280},
    "2 min": {"target_words": 290, "max_new_tokens": 520},
    "3 min": {"target_words": 430, "max_new_tokens": 760},
}
DEFAULT_LENGTH = "2 min"
MAX_CONTINUATIONS = 3  # extra passes to reach target length

_tokenizer: Any | None = None
_model: Any | None = None


def _get_model() -> tuple[Any, Any]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        token = os.environ.get("HF_TOKEN") or None
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            token=token,
            trust_remote_code=True,
        )
        _model.eval()
    return _tokenizer, _model


def generate_story(
    genre: str,
    mood: str,
    hero_name: str = "",
    language: str = "en",  # kept for signature compatibility; text is always English here
    length: str = DEFAULT_LENGTH,
    energy: float = 0.45,  # 0.0 = very calm, 1.0 = lively/energetic (for kids)
) -> str:
    """Generate a calm, meaningful English bedtime story that actually reaches the target length."""
    genre = _clean_choice(genre, VALID_GENRES, "genre")
    mood = _clean_choice(mood, VALID_MOODS, "mood")
    length = length if length in LENGTH_PRESETS else DEFAULT_LENGTH
    preset = LENGTH_PRESETS[length]
    hero_name = (hero_name or "").strip()[:40]
    target_words = preset["target_words"]

    energy = max(0.0, min(1.0, float(energy)))
    if energy >= 0.66:
        energy_line = ("Tell it with bright, playful energy — fun sounds, a little excitement and "
                       "wonder — so a child stays delighted, then settle into calm at the very end.")
    elif energy <= 0.33:
        energy_line = ("Tell it very softly and slowly throughout, soothing and dreamy, like easing "
                       "a child to sleep.")
    else:
        energy_line = ("Warm and gently lively, easing into a calm, sleepy ending.")

    user_prompt = (
        "Write ONLY the story text (no title, no labels).\n"
        f"Audience: a 6-year-old at bedtime.\n"
        f"Genre: {genre}. Mood: {mood}. Hero: {hero_name or 'a kind little child'}.\n"
        f"{energy_line}\n"
        f"Aim for about {target_words} words — a full story with a clear beginning, middle, and a "
        "soft sleepy ending.\n\n"
        "Story:"
    )

    tokenizer, model = _get_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    story = _generate_once(tokenizer, model, messages, preset["max_new_tokens"])
    story = _normalize_story(story)

    # Continue-until-target: small models under-shoot length; extend gently a few times.
    attempts = 0
    while _word_count(story) < int(target_words * 0.85) and attempts < MAX_CONTINUATIONS:
        attempts += 1
        cont_messages = messages + [
            {"role": "assistant", "content": story},
            {
                "role": "user",
                "content": (
                    "Continue the same story naturally — do not repeat earlier sentences and do not "
                    "start over. Add the next beat with vivid, gentle detail. If the story is nearly "
                    "complete, guide it toward a soft, sleepy good-night ending. Plain text only."
                ),
            },
        ]
        extra = _generate_once(tokenizer, model, cont_messages, max(160, preset["max_new_tokens"] // 2))
        extra = _normalize_story(extra)
        if extra and extra[:40].lower() not in story.lower():
            story = (story.rstrip() + " " + extra.lstrip()).strip()

    story = _trim_to_sentence(story, target_words)
    if not story:
        raise RuntimeError("The story model returned empty text. Please try again.")
    return story


def _generate_once(tokenizer, model, messages, max_new_tokens: int) -> str:
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
            max_new_tokens=max_new_tokens,
            min_new_tokens=int(max_new_tokens * 0.6),  # force a real length, not 2 sentences
            temperature=0.8,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def _clean_choice(value: str, allowed: set[str], label: str) -> str:
    cleaned = (value or "").strip().lower()
    if cleaned not in allowed:
        raise ValueError(f"Please choose a supported {label}: {', '.join(sorted(allowed))}.")
    return cleaned


def _word_count(text: str) -> int:
    return len(text.split())


def _normalize_story(text: str) -> str:
    story = re.sub(r"\s+", " ", text).strip().strip('"')

    # Strip markdown the TTS would otherwise read aloud.
    story = re.sub(r"\*{1,3}", "", story)
    story = re.sub(r"_{1,3}", "", story)
    story = re.sub(r"~{1,2}", "", story)
    story = re.sub(r"`{1,3}[^`]*`{1,3}", "", story)
    story = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", story)
    story = re.sub(r"[#]+ ", "", story)
    story = re.sub(r"^\s*[-•]\s+", "", story, flags=re.MULTILINE)

    # Strip echoed prompt scaffolding.
    for pat in [
        r"^(Once the story|Here is|Sure[,!]?|Story:|Genre:|Mood:|Hero:?|Title:|Write|Return|Language|Continue)[\s:]*",
    ]:
        story = re.sub(pat, "", story, count=1, flags=re.IGNORECASE)

    story = re.sub(r"\.{4,}", "...", story)
    story = re.sub(r"\.\s*\.\s*\.", "...", story)
    return story.strip()


def _trim_to_sentence(story: str, max_words: int) -> str:
    words = story.split()
    if len(words) <= int(max_words * 1.25):
        return story.strip()
    trimmed = " ".join(words[: int(max_words * 1.25)])
    last_end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if last_end > 80:
        return trimmed[: last_end + 1].strip()
    return trimmed.rstrip(" ,;:") + "."
