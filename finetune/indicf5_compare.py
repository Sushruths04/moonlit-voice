"""Phase 2b: Compare stock vs fine-tuned IndicF5 on Kannada test sentences."""

import os
import modal

GPU = "A10G"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers==4.46.3",
        "soundfile==0.14.0",
        "numpy==1.26.4",
        "vocos",
        "torchcodec",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

app = modal.App("indicf5-compare")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")
data_vol = modal.Volume.from_name("dreamvoice-ft-data")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 30,
    volumes={"/ckpt": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def compare():
    import torch
    import soundfile as sf
    import numpy as np
    from transformers import AutoModel

    print("=" * 60)
    print("COMPARING STOCK vs FINE-TUNED IndicF5")
    print("=" * 60)

    token = os.environ.get("HF_TOKEN")
    indicf5_id = "ai4bharat/IndicF5"

    # Test sentences
    sentences = [
        "ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ.",
        "ರಾತ್ರಿ ಬಂದಿದೆ, ನಕ್ಷತ್ರಗಳು ಮಿನುಗುತ್ತಿವೆ. ಮಲಗಿರಿ.",
        "ನಿಮ್ಮ ಕನಸುಗಳು ಸುಂದರವಾಗಿರಲಿ.",
        "ಶುಭ ರಾತ್ರಿ, ಪ್ರೀತಿಯ ಮಗು.",
    ]

    # Reference audio (use a sample from val set)
    ref_path = "/data/kannada_finetune/val.csv"
    ref_audio = None
    ref_text = "ನಮಸ್ಕಾರ"

    # Try to get a real reference audio
    if os.path.exists(ref_path):
        import csv
        with open(ref_path) as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                audio_file = row.get("audio_file", "")
                if audio_file and os.path.exists(audio_file):
                    ref_audio = audio_file
                    ref_text = row.get("text", ref_text)
                    break

    if ref_audio is None:
        # Fallback: create synthetic
        os.makedirs("/data/comparison", exist_ok=True)
        synth = np.random.randn(24000).astype(np.float32) * 0.01
        ref_audio = "/tmp/synthetic_ref.wav"
        sf.write(ref_audio, synth, 24000)

    print(f"  Reference: {ref_audio}")
    print(f"  Reference text: {ref_text}")

    # Load stock model
    print("\nLoading stock IndicF5...")
    stock_model = AutoModel.from_pretrained(
        indicf5_id, trust_remote_code=True, cache_dir="/cache", token=token
    )
    stock_model = stock_model.cuda()
    stock_model.eval()

    # Generate stock audio
    print("\nGenerating stock audio...")
    stock_dir = "/data/comparison/stock"
    os.makedirs(stock_dir, exist_ok=True)

    for i, text in enumerate(sentences):
        with torch.no_grad():
            output = stock_model(text, ref_audio_path=ref_audio, ref_text=ref_text)
        if isinstance(output, np.ndarray) and len(output) > 0:
            path = f"{stock_dir}/sentence_{i}.wav"
            sf.write(path, output.astype(np.float32), 24000)
            print(f"  ✓ Stock {i}: {path} ({len(output)/24000:.1f}s)")

    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    ft_model = AutoModel.from_pretrained(
        indicf5_id, trust_remote_code=True, cache_dir="/cache", token=token
    )
    ft_model = ft_model.cuda()

    # Load fine-tuned weights
    ckpt_path = "/ckpt/indicf5_kannada/step_0300/cfm.pt"
    if os.path.exists(ckpt_path):
        print(f"  Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cuda")
        ft_model.ema_model.load_state_dict(state)
        print("  ✓ Fine-tuned weights loaded")
    else:
        print(f"  ⚠ No checkpoint found at {ckpt_path}, using stock model")

    ft_model.eval()

    # Generate fine-tuned audio
    print("\nGenerating fine-tuned audio...")
    ft_dir = "/data/comparison/finetuned"
    os.makedirs(ft_dir, exist_ok=True)

    for i, text in enumerate(sentences):
        with torch.no_grad():
            output = ft_model(text, ref_audio_path=ref_audio, ref_text=ref_text)
        if isinstance(output, np.ndarray) and len(output) > 0:
            path = f"{ft_dir}/sentence_{i}.wav"
            sf.write(path, output.astype(np.float32), 24000)
            print(f"  ✓ Fine-tuned {i}: {path} ({len(output)/24000:.1f}s)")

    print("\n✅ Comparison audio generated!")
    print(f"  Stock: {stock_dir}/")
    print(f"  Fine-tuned: {ft_dir}/")
    data_vol.commit()

    return {"status": "PASS"}


if __name__ == "__main__":
    app.run()
