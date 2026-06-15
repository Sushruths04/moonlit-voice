"""Download and prepare LibriTTS train-clean-100 for VoxCPM2 fine-tuning.

LibriTTS has clean, single-speaker English speech at 24kHz.
We select one consistent female speaker for bedtime narration style.

Usage:
    modal run prepare_english_data.py
"""

from __future__ import annotations

import modal

APP_NAME = "dreamvoice-data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "wget", "tar")
    .pip_install(
        "soundfile==0.12.1",
        "librosa==0.10.2",
        "numpy==1.26.4",
    )
)

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,
    timeout=7200,
    volumes={"/data": data_vol},
)
def prepare_libritts():
    """Download LibriTTS train-clean-100 and prepare manifest for VoxCPM2."""
    import os
    import json
    import subprocess
    import soundfile as sf
    import numpy as np

    print("=== Downloading LibriTTS train-clean-100 ===")
    
    output_dir = "/data/english_libritts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already downloaded
    archive_path = "/tmp/train-clean-100.tar.gz"
    extract_dir = "/tmp/LibriTTS"
    
    if not os.path.exists(extract_dir):
        url = "http://www.openslr.org/resources/60/train-clean-100.tar.gz"
        print(f"Downloading from {url}...")
        subprocess.run(["wget", "-q", "-O", archive_path, url], check=True)
        
        print("Extracting...")
        subprocess.run(["tar", "-xzf", archive_path, "-C", "/tmp/"], check=True)
        print("✓ Downloaded and extracted")
    else:
        print("✓ Already extracted")
    
    # Walk through LibriTTS structure
    libritts_dir = os.path.join(extract_dir, "LibriTTS", "train-clean-100")
    
    manifest = []
    min_dur, max_dur = 2.0, 20.0
    target_sr = 24000
    
    processed = 0
    skipped = 0
    
    # Select a consistent female speaker (speaker 19 = clean female voice)
    # Common good female speakers: 19, 84, 121, 237
    target_speakers = {"19", "84", "121", "237"}
    
    for speaker_id in sorted(os.listdir(libritts_dir)):
        speaker_dir = os.path.join(libritts_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
            
        for chapter_id in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if not os.path.isdir(chapter_dir):
                continue
            
            # Find .wav files
            for fname in sorted(os.listdir(chapter_dir)):
                if not fname.endswith(".wav"):
                    continue
                    
                if processed >= 1000:  # Cap for first run
                    break
                
                try:
                    wav_path = os.path.join(chapter_dir, fname)
                    transcript_path = wav_path.replace(".wav", ".normalized.txt")
                    
                    # Read transcript
                    if not os.path.exists(transcript_path):
                        skipped += 1
                        continue
                    
                    with open(transcript_path) as f:
                        transcript = f.read().strip()
                    
                    if not transcript:
                        skipped += 1
                        continue
                    
                    # Read audio
                    waveform, sr = sf.read(wav_path, dtype="float32")
                    
                    # Resample if needed
                    if sr != target_sr:
                        import librosa
                        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                    
                    duration = len(waveform) / sr
                    
                    if duration < min_dur or duration > max_dur:
                        skipped += 1
                        continue
                    
                    # Save to output
                    out_fname = f"libritts_en_{processed:05d}.wav"
                    out_path = os.path.join(output_dir, out_fname)
                    sf.write(out_path, waveform, sr)
                    
                    manifest.append({
                        "audio": out_path,
                        "text": transcript,
                        "duration": duration,
                        "speaker": speaker_id,
                        "source": "libritts",
                    })
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed: {processed}, Skipped: {skipped}")
                        
                except Exception as e:
                    skipped += 1
                    continue
        
        if processed >= 1000:
            break
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n=== Results ===")
    print(f"Processed: {processed} clips")
    print(f"Skipped: {skipped} clips")
    print(f"Manifest: {manifest_path}")
    
    durations = [m["duration"] for m in manifest]
    print(f"Avg duration: {np.mean(durations):.1f}s")
    print(f"Total hours: {sum(durations)/3600:.2f}h")
    
    data_vol.commit()
    print("✓ Data saved to Modal volume")


if __name__ == "__main__":
    modal.run(prepare_libritts)
