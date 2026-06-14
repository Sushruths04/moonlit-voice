#!/usr/bin/env python3
"""Download StoryNory dataset and prepare VoxCPM2 fine-tuning manifests.

Downloads from HuggingFace (Pavankalyan/StoryNoryTTS), resamples to 16kHz mono,
filters to 3-15s clips, and writes JSONL manifests for VoxCPM2 training.

Usage:
    python finetune/download_storynory.py

Outputs:
    finetune/data/clean/*.wav (16k mono)
    finetune/data/manifest_train.jsonl
    finetune/data/manifest_val.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
CLEAN_DIR = HERE / "data" / "clean"
TRAIN_OUT = HERE / "data" / "manifest_train.jsonl"
VAL_OUT = HERE / "data" / "manifest_val.jsonl"

TARGET_SR = 16_000
MIN_SEC, MAX_SEC = 3.0, 15.0
VAL_FRACTION = 0.1
MAX_CLIPS = 5000  # use subset for fast experiments; set to None for full dataset


def _fail(msg: str) -> None:
    print(f"\n❌ {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    try:
        import librosa
        import soundfile as sf
        from datasets import load_dataset
    except ImportError:
        _fail("install deps: pip install datasets librosa soundfile numpy")

    print("📦 Downloading StoryNory dataset from HuggingFace...")
    ds = load_dataset("Pavankalyan/StoryNoryTTS", split="train", trust_remote_code=True)
    print(f"   Loaded {len(ds)} clips")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0
    for i, row in enumerate(ds):
        if MAX_CLIPS and len(records) >= MAX_CLIPS:
            break

        # Extract audio
        audio_data = row["audio"]
        if audio_data is None:
            skipped += 1
            continue

        # Get array and sampling rate
        if isinstance(audio_data, dict):
            array = audio_data.get("array")
            orig_sr = audio_data.get("sampling_rate", 22050)
        else:
            array = audio_data
            orig_sr = getattr(audio_data, "sampling_rate", 22050)

        if array is None:
            skipped += 1
            continue

        import numpy as np

        audio = np.array(array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono

        # Resample to 16kHz
        if orig_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

        dur = len(audio) / TARGET_SR
        if dur < MIN_SEC or dur > MAX_SEC:
            skipped += 1
            continue

        # Get transcript
        text = (row.get("text") or row.get("transcript") or "").strip()
        if not text:
            skipped += 1
            continue

        # Save clean WAV
        clip_id = f"sn_{i:06d}.wav"
        out_path = CLEAN_DIR / clip_id
        sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

        records.append({
            "audio": str(out_path),
            "text": text,
            "duration": round(dur, 2),
        })

        if len(records) % 500 == 0:
            print(f"   Processed {len(records)} clips ({skipped} skipped)...")

    if len(records) < 10:
        _fail(f"Only {len(records)} valid clips — need at least 10")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(records)

    n_val = max(1, int(len(records) * VAL_FRACTION))
    val, train = records[:n_val], records[n_val:]

    for path, rows in ((TRAIN_OUT, train), (VAL_OUT, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(r["duration"] for r in records)
    print(f"\n✅ {len(records)} clips ({total/60:.1f} min) → {len(train)} train / {len(val)} val")
    print(f"   {TRAIN_OUT}")
    print(f"   {VAL_OUT}")


if __name__ == "__main__":
    main()
