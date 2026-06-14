"""Fine-tune IndicF5 Kannada on 800 IndicTTS clips.

Uses CFM (ema_model) training like the original indicf5_finetune.py.
Runs on A100-80GB for speed.

Usage:
    modal run finetune/indicf5_finetune_v2.py
"""

import modal

app = modal.App("indicf5-kannada-ft-v2")
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
    image=image, gpu="A100-80GB", timeout=14400,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def train():
    import os, json
    import numpy as np
    import torch
    import soundfile as sf

    print("=" * 60)
    print("IndicF5 Kannada Fine-Tune v2 (800 IndicTTS clips)")
    print("=" * 60)

    # Load manifest
    manifest_path = "/data/kannada_tts/manifest.jsonl"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare_kannada_data.py first.")

    with open(manifest_path) as f:
        manifest = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(manifest)} clips")

    # Split 90/10
    np.random.seed(42)
    np.random.shuffle(manifest)
    split = int(len(manifest) * 0.9)
    train_data = manifest[:split]
    val_data = manifest[split:]

    # Save CSV for IndicF5 training format
    os.makedirs("/data/kannada_ft_v2", exist_ok=True)
    with open("/data/kannada_ft_v2/train.csv", "w") as f:
        f.write("audio_file|text\n")
        for m in train_data:
            f.write(f"{m['audio']}|{m['text']}\n")
    with open("/data/kannada_ft_v2/val.csv", "w") as f:
        f.write("audio_file|text\n")
        for m in val_data:
            f.write(f"{m['audio']}|{m['text']}\n")

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model
    print("\nLoading IndicF5...")
    from transformers import AutoModel
    token = os.environ.get("HF_TOKEN") or None
    model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5", trust_remote_code=True, cache_dir="/data/cache", token=token
    )
    model = model.cuda()

    cfm = model.ema_model
    print(f"CFM parameters: {sum(p.numel() for p in cfm.parameters()):,}")

    # Audio loading
    def load_audio(path, sr=24000):
        import torchaudio
        wav, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze()

    # Training
    print("\nStarting training (500 steps)...")
    cfm.train()
    optimizer = torch.optim.Adam(cfm.parameters(), lr=5e-6)
    batch_size = 2
    save_dir = "/ckpt/indicf5_kannada_v2"
    os.makedirs(save_dir, exist_ok=True)

    # Eval reference
    eval_ref = None
    eval_ref_text = ""
    for m in val_data:
        try:
            eval_ref = load_audio(m["audio"]).cuda()
            eval_ref_text = m["text"]
            break
        except:
            continue
    if eval_ref is None:
        eval_ref = torch.randn(24000).float().cuda() * 0.01

    eval_text = "ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ."

    losses = []
    for step in range(500):
        # Sample batch
        items = np.random.choice(train_data, size=batch_size, replace=True)
        waves, texts = [], []
        for item in items:
            try:
                wav = load_audio(item["audio"])
                waves.append(wav)
                texts.append(item["text"])
            except:
                continue

        if not waves:
            continue

        max_len = max(w.shape[0] for w in waves)
        padded = torch.zeros(len(waves), max_len)
        for i, w in enumerate(waves):
            padded[i, :w.shape[0]] = w
        padded = padded.cuda()

        optimizer.zero_grad()
        loss, _, _ = cfm(padded, texts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            avg = np.mean(losses[-10:]) if losses else loss.item()
            print(f"  Step {step:3d}: loss={loss.item():.4f}, avg={avg:.4f}")

        # Checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            ckpt_path = f"{save_dir}/step_{step+1:04d}"
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(cfm.state_dict(), f"{ckpt_path}/cfm.pt")
            with open(f"{ckpt_path}/config.json", "w") as f:
                json.dump({"step": step+1, "loss": float(np.mean(losses[-50:]))}, f)
            print(f"  ✓ Saved {ckpt_path}")

            # Quick eval
            cfm.eval()
            try:
                with torch.no_grad():
                    mel = cfm.mel_spec(eval_ref.unsqueeze(0))
                    mel = mel.permute(0, 2, 1)
                    out, _ = cfm.sample(
                        cond=mel, text=[eval_text],
                        duration=mel.shape[1] + 100,
                        steps=10, cfg_strength=1.0,
                    )
                    if out is not None:
                        audio_out = model.vocoder(out.permute(0, 2, 1))
                        audio_np = audio_out.squeeze().cpu().numpy()
                        sf.write(f"{ckpt_path}/eval.wav", audio_np, 24000)
                        print(f"  ✓ Eval: {len(audio_np)/24000:.1f}s")
            except Exception as e:
                print(f"  ⚠ Eval error: {e}")
            cfm.train()
            data_vol.commit()

    # Final save
    torch.save(cfm.state_dict(), f"{save_dir}/final/cfm.pt")
    data_vol.commit()

    print(f"\n{'='*60}")
    print(f"✅ DONE — Final loss: {np.mean(losses[-50:]):.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    modal.run(train)
