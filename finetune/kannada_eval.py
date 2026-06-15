"""Comprehensive Kannada TTS evaluation — stock vs all checkpoints.

Tests with ACTUAL user reference audio. Measures:
- Speaking rate (syllables/sec for Kannada)
- Spectral flatness (naturalness)
- Prosody variation (pitch std dev)
- Silence ratio
- RMS energy / peak / dynamic range
- Duration consistency
- MOS proxy (based on spectral + prosody features)
"""

import modal

app = modal.App("kannada-eval")
hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4", "librosa==0.10.2",
        "safetensors", "scipy",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

# 10 Kannada test sentences — bedtime story appropriate
TEST_SENTENCES = [
    ("ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ.", "Kids sleep. Moon waits for you."),
    ("ಒಂದು ಕಾಡಿನಲ್ಲಿ ಒಂದು ಚಿಕ್ಕ ಮರಿ ಹುಲಿ ವಾಸಿಸುತ್ತಿತ್ತು.", "A little tiger lived in a forest."),
    ("ರಾತ್ರಿ ಬಂದಾಗ ಆಕಾಶದಲ್ಲಿ ನಕ್ಷತ್ರಗಳು ಮಿನುಗುತ್ತವೆ.", "Stars twinkle at night."),
    ("ತಾಯಿ ಹುಲಿ ತನ್ನ ಮರಿಗಳಿಗೆ ಪ್ರೀತಿಯಿಂದ ಹಾಲು ಕುಡಿಸಿತು.", "Mother tiger fed milk to cubs."),
    ("ಚಿಕ್ಕ ಹುಲಿ ಪ್ರತಿದಿನ ಬೆಳಗ್ಗೆ ಎದ್ದು ಆಟವಾಡುತ್ತಿತ್ತು.", "Little tiger woke up daily and played."),
    ("ಆಕಾಶದಲ್ಲಿ ಬಿಳಿ ಮೋಡಗಳು ತೇಲುತ್ತಿದ್ದವು.", "White clouds floated in the sky."),
    ("ನದಿಯ ನೀರು ತಂಪಾಗಿ ಹರಿಯುತ್ತಿತ್ತು.", "River water flowed cool."),
    ("ಪಕ್ಷಿಗಳು ಮರದ ಮೇಲೆ ಹಾಡುತ್ತಿದ್ದವು.", "Birds sang on the tree."),
    ("ಮಳೆ ಬಂದಾಗ ಭೂಮಿ ಹಸಿರಾಯಿತು.", "Earth turned green when rain came."),
    ("ಮಕ್ಕಳು ಅಮ್ಮನ ಜೊತೆ ಮಲಗಿದರು.", "Children slept with mother."),
]


@app.function(
    image=image, gpu="A100-80GB", timeout=3600,
    volumes={"/cache": hf_cache, "/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def evaluate():
    import os, json, time
    import torch
    import numpy as np
    import soundfile as sf
    import librosa

    print("=" * 70)
    print("COMPREHENSIVE KANNADA TTS EVALUATION")
    print("=" * 70)

    token = os.environ.get("HF_TOKEN") or None
    INDICF5_ID = "ai4bharat/IndicF5"

    from transformers import AutoModel

    def load_audio(path, sr=24000):
        import torchaudio
        wav, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze()

    def analyze_audio(audio, sr):
        """Comprehensive audio analysis."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) == 0:
            return {}
        duration = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        dynamic_range = 20 * np.log10(peak / max(rms, 1e-10))

        # Silence ratio
        silence_thresh = 10 ** (-35.0 / 20.0)
        silence_pct = float(np.mean(np.abs(audio) < silence_thresh) * 100)

        # Spectral features
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        total_energy = np.sum(fft)
        if total_energy > 0:
            spectral_centroid = float(np.sum(freqs * fft) / total_energy)
            # Spectral flatness (0=noise, 1=tone)
            spectral_flatness = float(np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10))
        else:
            spectral_centroid = spectral_flatness = 0

        # Prosody: pitch analysis using librosa
        try:
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=sr)
            f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            if len(f0_clean) > 1:
                pitch_mean = float(np.mean(f0_clean))
                pitch_std = float(np.std(f0_clean))
                pitch_range = float(np.max(f0_clean) - np.min(f0_clean))
                # Voiced fraction
                voiced_pct = float(np.mean(voiced_flag) * 100) if voiced_flag is not None else 0
            else:
                pitch_mean = pitch_std = pitch_range = voiced_pct = 0
        except:
            pitch_mean = pitch_std = pitch_range = voiced_pct = 0

        # Speaking rate proxy: count energy peaks (approximate syllables)
        # Smooth the energy envelope
        hop = int(0.01 * sr)  # 10ms hop
        energy = np.array([np.mean(audio[i:i+hop]**2) for i in range(0, len(audio)-hop, hop)])
        if len(energy) > 0:
            # Count peaks above threshold
            peaks = np.diff((energy > np.mean(energy) * 0.3).astype(int))
            n_peaks = np.sum(peaks > 0)
            syllables_per_sec = n_peaks / duration if duration > 0 else 0
        else:
            syllables_per_sec = 0

        # MOS proxy: combination of features
        # Higher pitch variation = more natural
        # Lower silence = better
        # Moderate spectral flatness = natural speech
        # Speaking rate 4-6 syll/sec = natural
        mos_score = 0
        if pitch_std > 20: mos_score += 1
        if pitch_std > 40: mos_score += 1
        if silence_pct < 40: mos_score += 1
        if silence_pct < 25: mos_score += 1
        if 3 < syllables_per_sec < 7: mos_score += 1
        if 0.3 < spectral_flatness < 0.8: mos_score += 1
        if dynamic_range > 10: mos_score += 1

        return {
            "duration": round(duration, 2),
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "dynamic_range_db": round(dynamic_range, 1),
            "silence_pct": round(silence_pct, 1),
            "spectral_centroid_hz": round(spectral_centroid, 0),
            "spectral_flatness": round(spectral_flatness, 4),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "pitch_range_hz": round(pitch_range, 1),
            "voiced_pct": round(voiced_pct, 1),
            "syll_per_sec": round(syllables_per_sec, 1),
            "mos_proxy": mos_score,
        }

    # ─── Prepare user reference audio ──────────────────────────
    print("\n[1] Preparing user reference audio...")
    user_ref_path = "/tmp/user_ref.wav"
    # Load from volume
    ref_audio, ref_sr = sf.read("/data/reference_voice.wav")
    # Resample to 24kHz if needed
    if ref_sr != 24000:
        import torchaudio
        ref_tensor = torch.from_numpy(ref_audio).float()
        if ref_tensor.dim() == 1:
            ref_tensor = ref_tensor.unsqueeze(0)
        ref_tensor = torchaudio.functional.resample(ref_tensor, ref_sr, 24000)
        ref_audio = ref_tensor.squeeze().numpy()
    sf.write(user_ref_path, ref_audio, 24000)
    ref_text = "ನಮಸ್ಕಾರ ನನ್ನ ಹೆಸರು ಪ್ರಿಯಾ ನಾನು ನಿಮ್ಮ ಮಕ್ಕಳಿಗೆ ಕಥೆ ಹೇಳುತ್ತೇನೆ"
    print(f"  ✓ User ref: {len(ref_audio)/24000:.1f}s")

    # Also get a training data reference for comparison
    train_ref_path = None
    train_ref_text = ""
    manifest_path = "/data/kannada_tts/manifest.jsonl"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for item in lines[:30]:
            try:
                audio_path = item["audio"]
                # Try volume path
                if os.path.exists(audio_path):
                    wav = load_audio(audio_path)
                    sf.write("/tmp/train_ref.wav", wav.numpy(), 24000)
                    train_ref_path = "/tmp/train_ref.wav"
                    train_ref_text = item["text"]
                    print(f"  ✓ Train ref: {train_ref_text[:50]}...")
                    break
            except:
                continue
    if train_ref_path is None:
        print("  ⚠ No train ref found, using user ref for both")

    # ─── Load base model ───────────────────────────────────────
    print("\n[2] Loading base IndicF5...")
    model = AutoModel.from_pretrained(
        INDICF5_ID, trust_remote_code=True, cache_dir="/cache", token=token)
    try:
        model = model.to("cuda")
    except:
        pass
    print("  ✓ Base model ready")

    # ─── Define checkpoint configs ─────────────────────────────
    checkpoints = {
        "stock": None,
        "v3_step0200": "/ckpt/indicf5_kannada_v3/step_0200/cfm.pt",
        "v3_step0400": "/ckpt/indicf5_kannada_v3/step_0400/cfm.pt",
        "v3_step0600": "/ckpt/indicf5_kannada_v3/step_0600/cfm.pt",
        "v3_step0800": "/ckpt/indicf5_kannada_v3/step_0800/cfm.pt",
        "v3_final": "/ckpt/indicf5_kannada_v3/final/cfm.pt",
        "v2_step0500": "/ckpt/indicf5_kannada_v2/step_0500/cfm.pt",
    }

    # ─── Run evaluation ────────────────────────────────────────
    all_results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING: {ckpt_name}")
        print(f"{'='*70}")

        # Load checkpoint
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.ema_model.load_state_dict(state)
            print(f"  ✓ Loaded {ckpt_path}")
        elif ckpt_path:
            print(f"  ✗ NOT FOUND: {ckpt_path}")
            continue
        else:
            print(f"  Using stock weights (no checkpoint)")

        ckpt_results = {"user_ref": [], "train_ref": []}

        for ref_label, ref_path, ref_txt in [
            ("user_ref", user_ref_path, ref_text),
            ("train_ref", train_ref_path, train_ref_text),
        ]:
            if ref_path is None:
                continue

            print(f"\n  --- Reference: {ref_label} ---")
            for i, (kannada, english) in enumerate(TEST_SENTENCES):
                try:
                    t0 = time.time()
                    with torch.no_grad():
                        out = model(kannada, ref_audio_path=ref_path, ref_text=ref_txt)
                    gen_time = time.time() - t0

                    audio = np.asarray(out, dtype=np.float32).flatten()
                    if audio.max() > 1.0:
                        audio = audio / 32768.0

                    metrics = analyze_audio(audio, 24000)
                    metrics["gen_time"] = round(gen_time, 2)
                    metrics["sentence_en"] = english

                    # Save WAV
                    out_dir = f"/data/eval_{ckpt_name}_{ref_label}"
                    os.makedirs(out_dir, exist_ok=True)
                    sf.write(f"{out_dir}/sent_{i:02d}.wav", audio, 24000)

                    ckpt_results[ref_label].append(metrics)

                    status = "✓" if metrics["mos_proxy"] >= 4 else "~" if metrics["mos_proxy"] >= 2 else "✗"
                    print(f"    {status} [{i:2d}] {metrics['duration']:5.1f}s "
                          f"sil={metrics['silence_pct']:4.1f}% "
                          f"pitch={metrics['pitch_std_hz']:5.1f}Hz "
                          f"syll/s={metrics['syll_per_sec']:4.1f} "
                          f"MOS={metrics['mos_proxy']}/7 "
                          f"gen={gen_time:.1f}s")

                except Exception as e:
                    print(f"    ✗ [{i:2d}] FAILED: {e}")

        # Summary for this checkpoint
        for ref_label in ["user_ref", "train_ref"]:
            results = ckpt_results.get(ref_label, [])
            if not results:
                continue
            avg = {
                "silence_pct": np.mean([r["silence_pct"] for r in results]),
                "pitch_std": np.mean([r["pitch_std_hz"] for r in results]),
                "syll_per_sec": np.mean([r["syll_per_sec"] for r in results]),
                "mos": np.mean([r["mos_proxy"] for r in results]),
                "duration": np.mean([r["duration"] for r in results]),
                "spectral_flatness": np.mean([r["spectral_flatness"] for r in results]),
            }
            print(f"\n  SUMMARY ({ref_label}):")
            print(f"    Silence: {avg['silence_pct']:.1f}%")
            print(f"    Pitch variation: {avg['pitch_std']:.1f}Hz")
            print(f"    Speaking rate: {avg['syll_per_sec']:.1f} syll/s")
            print(f"    MOS proxy: {avg['mos']:.1f}/7")
            print(f"    Avg duration: {avg['duration']:.1f}s")
            print(f"    Spectral flatness: {avg['spectral_flatness']:.4f}")

        all_results[ckpt_name] = ckpt_results

    # ─── Final comparison table ────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE (user reference)")
    print("=" * 70)
    print(f"{'Checkpoint':20s} {'Silence':>8s} {'Pitch Std':>10s} {'Syll/s':>7s} {'MOS':>5s} {'Dur':>6s} {'Flat':>7s}")
    print("-" * 70)

    for ckpt_name in checkpoints:
        results = all_results.get(ckpt_name, {}).get("user_ref", [])
        if not results:
            continue
        avg_sil = np.mean([r["silence_pct"] for r in results])
        avg_pitch = np.mean([r["pitch_std_hz"] for r in results])
        avg_syll = np.mean([r["syll_per_sec"] for r in results])
        avg_mos = np.mean([r["mos_proxy"] for r in results])
        avg_dur = np.mean([r["duration"] for r in results])
        avg_flat = np.mean([r["spectral_flatness"] for r in results])
        print(f"{ckpt_name:20s} {avg_sil:7.1f}% {avg_pitch:9.1f}Hz {avg_syll:6.1f} {avg_mos:5.1f} {avg_dur:5.1f}s {avg_flat:7.4f}")

    # Save full results
    with open("/data/eval_results.json", "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    data_vol.commit()
    print("\n✅ EVALUATION COMPLETE — saved to /data/eval_results.json")


if __name__ == "__main__":
    modal.run(evaluate)
