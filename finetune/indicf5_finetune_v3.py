"""Fine-tune IndicF5 Kannada v3 — 1000 steps, lower LR, better monitoring.

Key changes from v2:
- 1000 steps (vs 500)
- LR 2e-6 (vs 5e-6) — more conservative
- Save every 200 steps for comparison
- Better loss tracking
"""

import modal

app = modal.App("indicf5-kannada-ft-v3")
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
    image=image, gpu="A100-80GB", timeout=21600,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def train():
    import os, json
    import numpy as np
    import torch
    import soundfile as sf

    print("=" * 60)
    print("IndicF5 Kannada Fine-Tune v3 (1000 steps, 800 clips)")
    print("=" * 60)

    # Load manifest
    manifest_path = "/data/kannada_tts/manifest.jsonl"
    with open(manifest_path) as f:
        manifest = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(manifest)} clips")

    # Split 90/10
    np.random.seed(42)
    np.random.shuffle(manifest)
    split = int(len(manifest) * 0.9)
    train_data = manifest[:split]
    val_data = manifest[split:]

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
    print("\nStarting training (1000 steps, LR 2e-6)...")
    cfm.train()
    optimizer = torch.optim.Adam(cfm.parameters(), lr=2e-6)
    batch_size = 2
    save_dir = "/ckpt/indicf5_kannada_v3"
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

    losses = []
    for step in range(1000):
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
            print(f"  Step {step:4d}: loss={loss.item():.4f}, avg={avg:.4f}")

        # Checkpoint every 200 steps
        if (step + 1) % 200 == 0:
            ckpt_path = f"{save_dir}/step_{step+1:04d}"
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(cfm.state_dict(), f"{ckpt_path}/cfm.pt")
            with open(f"{ckpt_path}/config.json", "w") as f:
                json.dump({"step": step+1, "loss": float(np.mean(losses[-50:]))}, f)
            print(f"  ✓ Saved {ckpt_path} (avg loss: {np.mean(losses[-50:]):.4f})")
            data_vol.commit()

    # Final save
    final_dir = f"{save_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    torch.save(cfm.state_dict(), f"{final_dir}/cfm.pt")
    data_vol.commit()

    print(f"\n{'='*60}")
    print(f"✅ DONE — Final loss: {np.mean(losses[-50:]):.4f}")
    print(f"Saved to: {save_dir}")
    print(f"Loss history: first={losses[0]:.4f}, last10_avg={np.mean(losses[-10:]):.4f}")


if __name__ == "__main__":
    modal.run(train)
