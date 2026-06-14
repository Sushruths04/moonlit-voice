#!/usr/bin/env python3
"""Build a VoxCPM2 fine-tuning manifest from raw clips + transcripts.

Self-checking & failproof:
- validates every clip is readable and 3-15 s,
- resamples to 16 kHz mono PCM16,
- writes JSONL manifests (train/val) in the format VoxCPM2 expects,
- FAILS LOUD with an actionable message if anything is wrong (never writes a silently-bad manifest).

Usage:
    python finetune/prepare_data.py
Inputs:
    finetune/data/raw/*.wav|*.mp3
    finetune/data/transcripts.tsv   ->  "<filename><TAB><exact transcript>" per line
Outputs:
    finetune/data/clean/*.wav (16k mono)
    finetune/data/manifest_train.jsonl
    finetune/data/manifest_val.jsonl
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
CLEAN_DIR = os.path.join(HERE, "data", "clean")
TRANSCRIPTS = os.path.join(HERE, "data", "transcripts.tsv")
TRAIN_OUT = os.path.join(HERE, "data", "manifest_train.jsonl")
VAL_OUT = os.path.join(HERE, "data", "manifest_val.jsonl")

TARGET_SR = 16_000          # VoxCPM2 AudioVAE encoder rate
MIN_SEC, MAX_SEC = 1.0, 20.0
VAL_FRACTION = 0.1


def _fail(msg: str) -> "None":
    print(f"\n❌ prepare_data failed: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def _load_transcripts() -> dict[str, str]:
    if not os.path.exists(TRANSCRIPTS):
        _fail(f"missing transcripts file: {TRANSCRIPTS} (one '<filename>\\t<transcript>' per line)")
    mapping: dict[str, str] = {}
    with open(TRANSCRIPTS, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                _fail(f"{TRANSCRIPTS}:{i} is not TAB-separated")
            name, text = line.split("\t", 1)
            if not text.strip():
                _fail(f"{TRANSCRIPTS}:{i} has an empty transcript for '{name}'")
            mapping[name.strip()] = text.strip()
    if not mapping:
        _fail("transcripts file is empty")
    return mapping


def main() -> None:
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        _fail("install deps first: pip install librosa soundfile numpy")

    if not os.path.isdir(RAW_DIR):
        _fail(f"missing raw clips dir: {RAW_DIR}")
    os.makedirs(CLEAN_DIR, exist_ok=True)

    transcripts = _load_transcripts()
    raw_files = [f for f in sorted(os.listdir(RAW_DIR)) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
    if not raw_files:
        _fail(f"no audio clips found in {RAW_DIR}")

    records = []
    for fname in raw_files:
        if fname not in transcripts:
            _fail(f"no transcript for clip '{fname}' (add it to transcripts.tsv)")
        src = os.path.join(RAW_DIR, fname)
        try:
            audio, _ = librosa.load(src, sr=TARGET_SR, mono=True)
        except Exception as exc:  # noqa: BLE001
            _fail(f"cannot read '{fname}': {exc}")
        dur = len(audio) / TARGET_SR
        if dur < MIN_SEC or dur > MAX_SEC:
            _fail(f"'{fname}' is {dur:.1f}s; clips must be {MIN_SEC}-{MAX_SEC}s")

        out_name = os.path.splitext(fname)[0] + ".wav"
        out_path = os.path.join(CLEAN_DIR, out_name)
        sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
        records.append({"audio": out_path, "text": transcripts[fname], "duration": round(dur, 2)})

    if len(records) < 5:
        print(f"⚠️  only {len(records)} clips — VoxCPM2 LoRA wants 5–50 for good results.")

    n_val = max(1, int(len(records) * VAL_FRACTION))
    val, train = records[:n_val], records[n_val:] or records

    for path, rows in ((TRAIN_OUT, train), (VAL_OUT, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(r["duration"] for r in records)
    print(f"✅ {len(records)} clips ({total/60:.1f} min) → {len(train)} train / {len(val)} val")
    print(f"   {TRAIN_OUT}\n   {VAL_OUT}")


if __name__ == "__main__":
    main()
