"""End-to-end backend smoke test for DreamVoice."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_utils import prepare_reference  # noqa: E402
from story import generate_story  # noqa: E402
from tts import clone_and_speak  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DreamVoice backend smoke test.")
    parser.add_argument("--reference-wav", help="Optional local speech clip to use as the voice reference.")
    parser.add_argument("--genre", default="dragons")
    parser.add_argument("--mood", default="funny")
    parser.add_argument("--hero-name", default="Maya")
    args = parser.parse_args()

    generated_ref = None
    cleaned_ref = None
    try:
        source_ref = args.reference_wav or _create_disposable_reference()
        if not args.reference_wav:
            generated_ref = source_ref

        cleaned_ref = prepare_reference(source_ref)
        story = generate_story(args.genre, args.mood, args.hero_name)
        out_wav = clone_and_speak(cleaned_ref, story)

        if not os.path.exists(out_wav) or os.path.getsize(out_wav) <= 1_000:
            raise RuntimeError("TTS output WAV is missing or too small.")

        print(out_wav)
    finally:
        for path in (cleaned_ref, generated_ref):
            if path:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


def _create_disposable_reference() -> str:
    ref_fn = modal.Function.from_name("dreamvoice-tts", "synthesize_reference")
    ref_bytes = ref_fn.remote(
        "Hello little moon, this is a gentle bedtime voice for testing DreamVoice. "
        "I am reading slowly and clearly in a quiet room, with a warm and calm voice. "
        "The stars are soft, the blanket is cozy, and it is nearly time to sleep."
    )
    fd, path = tempfile.mkstemp(prefix="dreamvoice_smoke_ref_", suffix=".wav")
    os.close(fd)
    with open(path, "wb") as ref_file:
        ref_file.write(ref_bytes)
    return path


if __name__ == "__main__":
    main()
