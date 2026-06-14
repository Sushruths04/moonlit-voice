"""Compare stock IndicF5 vs v2 fine-tuned on Kannada.

Downloads both models and generates comparison audio on Modal A100.
"""

import modal

app = modal.App("indicf5-compare-v2")
data_vol = modal.Volume.from_name("dreamvoice-ft-data")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4",
        "safetensors", "accelerate", "hydra-core", "omegaconf",
        "einops", "torchdiffeq", "vocos", "x_transformers",
        "ema_pytorch", "librosa",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)


@app.function(
    image=image, gpu="A100-80GB", timeout=1800,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def compare():
    import os
    import torch
    import soundfile as sf
    import numpy as np

    print("=" * 60)
    print("IndicF5 Stock vs v2 Fine-Tuned Comparison")
    print("=" * 60)

    # Load model
    from transformers import AutoModel
    token = os.environ.get("HF_TOKEN") or None

    print("Loading IndicF5...")
    model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5", trust_remote_code=True, cache_dir="/data/cache", token=token
    )
    model = model.cuda()
    print("✓ Model loaded")

    # Test texts
    test_texts = [
        "ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ.",
        "ಒಂದು ಕಾಡಿನಲ್ಲಿ ಒಂದು ಚಿಕ್ಕ ಮರಿ ಹುಲಿ ವಾಸಿಸುತ್ತಿತ್ತು.",
        "ರಾತ್ರಿ ಬಂದಾಗ ಆಕಾಶದಲ್ಲಿ ನಕ್ಷತ್ರಗಳು ಮಿನುಗುತ್ತವೆ.",
    ]

    # Use a reference audio from the training data
    ref_audio_path = None
    ref_text = ""
    manifest_path = "/data/kannada_tts/manifest.jsonl"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                import json
                item = json.loads(line)
                try:
                    wav, sr = torchaudio.load(item["audio"])
                    if sr != 24000:
                        wav = torchaudio.functional.resample(wav, sr, 24000)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    ref_audio_path = item["audio"]
                    ref_text = item["text"]
                    print(f"Using reference: {ref_text[:50]}...")
                    break
                except:
                    continue

    if ref_audio_path is None:
        print("No reference audio found, using synthetic")
        ref_audio = torch.randn(24000).float().cuda() * 0.01
        ref_text = "ನಮಸ್ಕಾರ"
    else:
        import torchaudio
        ref_audio, sr = torchaudio.load(ref_audio_path)
        if sr != 24000:
            ref_audio = torchaudio.functional.resample(ref_audio, sr, 24000)
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
        ref_audio = ref_audio.cuda()

    # Generate stock
    print("\n--- Stock IndicF5 ---")
    stock_dir = "/data/comparison_v2/stock"
    os.makedirs(stock_dir, exist_ok=True)

    for i, text in enumerate(test_texts):
        try:
            out = model(
                text,
                ref_audio_path=ref_audio_path if ref_audio_path else None,
                ref_text=ref_text,
            )
            audio = out.squeeze().cpu().numpy()
            path = f"{stock_dir}/stock_{i}.wav"
            sf.write(path, audio, 24000)
            print(f"  ✓ Stock {i}: {len(audio)/24000:.1f}s")
        except Exception as e:
            print(f"  ✗ Stock {i} failed: {e}")

    # Load fine-tuned v2
    print("\n--- Fine-tuned v2 ---")
    ft_dir = "/data/comparison_v2/finetuned"
    os.makedirs(ft_dir, exist_ok=True)

    # Load CFM checkpoint
    ckpt_path = "/ckpt/indicf5_kannada_v2/final/cfm.pt"
    if not os.path.exists(ckpt_path):
        # Try step_0500
        ckpt_path = "/ckpt/indicf5_kannada_v2/step_0500/cfm.pt"

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        cfm_state = torch.load(ckpt_path, map_location="cuda")
        model.ema_model.load_state_dict(cfm_state)
        print("✓ Checkpoint loaded")

        for i, text in enumerate(test_texts):
            try:
                out = model(
                    text,
                    ref_audio_path=ref_audio_path if ref_audio_path else None,
                    ref_text=ref_text,
                )
                audio = out.squeeze().cpu().numpy()
                path = f"{ft_dir}/ft_v2_{i}.wav"
                sf.write(path, audio, 24000)
                print(f"  ✓ FT v2 {i}: {len(audio)/24000:.1f}s")
            except Exception as e:
                print(f"  ✗ FT v2 {i} failed: {e}")
    else:
        print(f"✗ No checkpoint found at {ckpt_path}")
        print(f"  Available: {os.listdir('/ckpt/indicf5_kannada_v2/')}")

    # Copy to local
    import subprocess
    os.makedirs("/data/comparison_v2", exist_ok=True)
    print(f"\n✓ Comparison files in /data/comparison_v2/")
    print("  Stock: stock/stock_0.wav, stock_1.wav, stock_2.wav")
    print("  Fine-tuned: finetuned/ft_v2_0.wav, ft_v2_1.wav, ft_v2_2.wav")

    return {"status": "done"}


if __name__ == "__main__":
    modal.run(compare)
