"""Generate Kannada samples for analysis — stock vs v2 fine-tuned.

Runs on Modal A10G, downloads results locally for analysis.
"""

import modal
import io

app = modal.App("kannada-live-test")
hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)

f5_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4",
        "safetensors", "huggingface_hub",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

TEST_SENTENCES = [
    ("ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ.", "Kids, go to sleep. The moon is waiting for you."),
    ("ಒಂದು ಕಾಡಿನಲ್ಲಿ ಒಂದು ಚಿಕ್ಕ ಮರಿ ಹುಲಿ ವಾಸಿಸುತ್ತಿತ್ತು.", "A little tiger cub lived in a forest."),
    ("ರಾತ್ರಿ ಬಂದಾಗ ಆಕಾಶದಲ್ಲಿ ನಕ್ಷತ್ರಗಳು ಮಿನುಗುತ್ತವೆ.", "When night comes, stars twinkle in the sky."),
    ("ತಾಯಿ ಹುಲಿ ತನ್ನ ಮರಿಗಳಿಗೆ ಪ್ರೀತಿಯಿಂದ ಹಾಲು ಕುಡಿಸಿತು.", "The mother tiger lovingly fed milk to her cubs."),
    ("ಚಿಕ್ಕ ಹುಲಿ ಪ್ರತಿದಿನ ಬೆಳಗ್ಗೆ ಎದ್ದು ಆಟವಾಡುತ್ತಿತ್ತು.", "The little tiger woke up every morning and played."),
]


@app.function(
    image=f5_image, gpu="A10G", timeout=600,
    volumes={"/cache": hf_cache, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def generate_samples():
    import os
    import torch
    import soundfile as sf
    import numpy as np
    import json

    print("=" * 60)
    print("KANNADA LIVE QUALITY TEST")
    print("=" * 60)

    token = os.environ.get("HF_TOKEN") or None
    INDICF5_ID = "ai4bharat/IndicF5"
    V2_REPO = "mitvho09/IndicF5-Kannada-Bedtime-v2"

    # ─── Load model ─────────────────────────────────────────────
    print("\n[1] Loading base IndicF5...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        INDICF5_ID, trust_remote_code=True, cache_dir="/cache", token=token)
    print("  ✓ Base model loaded")

    # Load v2 weights
    try:
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id=V2_REPO, filename="model.safetensors",
            cache_dir="/cache", token=token)
        state = load_file(ckpt_path, device="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  ✓ v2 fine-tuned weights loaded ({len(state)} params)")
    except Exception as e:
        print(f"  ⚠ v2 load failed: {e}")

    model = model.cuda()
    print("  ✓ Model on GPU")

    # ─── Get reference audio ────────────────────────────────────
    print("\n[2] Getting reference audio...")
    ref_audio_path = None
    ref_text = ""
    manifest_path = "/data/kannada_tts/manifest.jsonl"

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for item in lines[:50]:
            try:
                audio_path = item["audio"]
                if os.path.exists(audio_path):
                    ref_audio_path = audio_path
                    ref_text = item["text"]
                    print(f"  ✓ Reference: {ref_text[:60]}...")
                    break
            except:
                continue

    if ref_audio_path is None:
        # Create a synthetic reference file
        import torchaudio
        ref_audio_path = "/tmp/synthetic_ref.wav"
        synthetic = torch.randn(1, 24000).float() * 0.01
        torchaudio.save(ref_audio_path, synthetic, 24000)
        ref_text = "ನಮಸ್ಕಾರ"
        print(f"  ⚠ Using synthetic reference")

    # ─── Generate samples ───────────────────────────────────────
    print(f"\n[3] Generating {len(TEST_SENTENCES)} samples...")
    os.makedirs("/data/kannada_live_v2", exist_ok=True)
    results = []

    for i, (kannada, english) in enumerate(TEST_SENTENCES):
        print(f"\n  --- Sample {i+1}: {english} ---")
        try:
            import time
            start = time.time()
            with torch.no_grad():
                audio_out = model(
                    kannada,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                )
            elapsed = time.time() - start

            audio = np.asarray(audio_out, dtype=np.float32).flatten()
            if audio.max() > 1.0:
                audio = audio / 32768.0

            duration = len(audio) / 24000

            # Save
            out_path = f"/data/kannada_live_v2/sample_{i+1}.wav"
            os.makedirs("/data/kannada_live_v2", exist_ok=True)
            sf.write(out_path, audio, 24000)

            # Analyze
            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            silence_thresh = 10 ** (-35.0 / 20.0)
            silence_ratio = float(np.mean(np.abs(audio) < silence_thresh))

            results.append({
                "id": i + 1,
                "kannada": kannada,
                "english": english,
                "duration_s": round(duration, 2),
                "gen_time_s": round(elapsed, 2),
                "rms": round(rms, 4),
                "peak": round(peak, 4),
                "silence_pct": round(silence_ratio * 100, 1),
            })

            print(f"    Duration: {duration:.1f}s | Gen: {elapsed:.1f}s | "
                  f"RMS: {rms:.4f} | Peak: {peak:.4f} | Silence: {silence_ratio*100:.1f}%")
            print(f"    Saved: {out_path}")

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            results.append({"id": i+1, "error": str(e)})

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_dur = np.mean([r["duration_s"] for r in valid])
        avg_rms = np.mean([r["rms"] for r in valid])
        avg_silence = np.mean([r["silence_pct"] for r in valid])
        avg_gen = np.mean([r["gen_time_s"] for r in valid])
        print(f"  Samples generated: {len(valid)}/{len(TEST_SENTENCES)}")
        print(f"  Avg duration: {avg_dur:.1f}s")
        print(f"  Avg gen time: {avg_gen:.1f}s")
        print(f"  Avg RMS energy: {avg_rms:.4f}")
        print(f"  Avg silence: {avg_silence:.1f}%")

    # Save results JSON
    with open("/data/kannada_live_v2/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to /data/kannada_live_v2/results.json")

    data_vol.commit()
    print("\n✅ DONE")

    return results


if __name__ == "__main__":
    modal.run(generate_samples)
