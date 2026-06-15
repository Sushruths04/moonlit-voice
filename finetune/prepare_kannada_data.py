"""Download and prepare Kannada TTS data from open datasets.

Uses:
1. SPRINGLab/IndicTTS_Kannada - 7.35h studio quality
2. ai4bharat/Rasa Kannada - 27h expressive speech

Usage:
    modal run prepare_kannada_data.py
"""

from __future__ import annotations

import modal

APP_NAME = "dreamvoice-data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "datasets==2.21.0",
        "soundfile==0.12.1",
        "librosa==0.10.2",
        "numpy==1.26.4",
        "huggingface_hub==1.19.0",
    )
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,
    timeout=3600,
    volumes={"/cache": hf_cache, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def prepare_kannada():
    """Download open Kannada TTS datasets and prepare manifest."""
    import os
    import json
    import soundfile as sf
    import numpy as np
    from datasets import load_dataset

    output_dir = "/data/kannada_tts"
    os.makedirs(output_dir, exist_ok=True)
    
    manifest = []
    min_dur, max_dur = 1.5, 20.0
    target_sr = 24000
    processed = 0
    
    # === 1. SPRINGLab/IndicTTS_Kannada (studio quality, 7.35h) ===
    print("=== Loading SPRINGLab/IndicTTS_Kannada ===")
    try:
        ds = load_dataset(
            "SPRINGLab/IndicTTS_Kannada",
            split="train",
            cache_dir="/cache/hf_datasets",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"  IndicTTS_Kannada: {len(ds)} clips")
        
        for item in ds:
            if processed >= 800:
                break
            try:
                audio = item["audio"]
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                
                if sr != target_sr:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                
                duration = len(waveform) / sr
                if duration < min_dur or duration > max_dur:
                    continue
                
                transcript = item.get("transcription", item.get("text", "")).strip()
                if not transcript:
                    continue
                
                filename = f"indicfts_kn_{processed:05d}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, waveform, sr)
                
                manifest.append({
                    "audio": filepath,
                    "text": transcript,
                    "duration": duration,
                    "source": "indicfts",
                })
                processed += 1
                
            except Exception:
                continue
        
        print(f"  Processed from IndicTTS: {processed}")
    except Exception as e:
        print(f"  IndicTTS failed: {e}")
    
    # === 2. ai4bharat/Rasa Kannada (27h expressive) ===
    print("\n=== Loading ai4bharat/Rasa Kannada ===")
    try:
        ds = load_dataset(
            "ai4bharat/Rasa",
            "kn",
            split="train",
            cache_dir="/cache/hf_datasets",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"  Rasa Kannada: {len(ds)} clips")
        
        count = 0
        for item in ds:
            if count >= 500:  # Cap at 500 from Rasa
                break
            if processed >= 1000:  # Total cap
                break
            try:
                audio = item["audio"]
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                
                if sr != target_sr:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                
                duration = len(waveform) / sr
                if duration < min_dur or duration > max_dur:
                    continue
                
                transcript = item.get("text", item.get("transcription", "")).strip()
                if not transcript:
                    continue
                
                filename = f"rasa_kn_{processed:05d}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, waveform, sr)
                
                manifest.append({
                    "audio": filepath,
                    "text": transcript,
                    "duration": duration,
                    "source": "rasa",
                })
                processed += 1
                count += 1
                
            except Exception:
                continue
        
        print(f"  Processed from Rasa: {count}")
    except Exception as e:
        print(f"  Rasa failed: {e}")
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n=== Results ===")
    print(f"Total clips: {processed}")
    print(f"Manifest: {manifest_path}")
    
    durations = [m["duration"] for m in manifest]
    print(f"Avg duration: {np.mean(durations):.1f}s")
    print(f"Total hours: {sum(durations)/3600:.2f}h")
    
    data_vol.commit()
    print("✓ Data saved to Modal volume")


if __name__ == "__main__":
    modal.run(prepare_kannada)
