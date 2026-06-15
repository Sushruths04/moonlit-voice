"""IndicF5 Kannada LoRA Fine-Tune v4 — rank 16, lr 1e-5, 2000 steps.

Patches Linear layers with LoRA by wrapping their forward methods.
More reliable than hooks for gradient flow.
"""

import modal

app = modal.App("indicf5-kannada-lora-v4")
data_vol = modal.Volume.from_name("dreamvoice-ft-data")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4",
        "safetensors", "accelerate",
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
    import torch.nn as nn
    import soundfile as sf

    print("=" * 60)
    print("IndicF5 Kannada LoRA Fine-Tune v4")
    print("  Rank: 16, LR: 1e-5, Steps: 2000")
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
    total_params = sum(p.numel() for p in cfm.parameters())
    print(f"CFM parameters: {total_params:,}")

    # Apply LoRA by patching forward methods
    RANK = 16
    ALPHA = 16

    class LoRALayer(nn.Module):
        def __init__(self, original_linear, rank=RANK, alpha=ALPHA):
            super().__init__()
            self.original = original_linear
            in_features = original_linear.in_features
            out_features = original_linear.out_features

            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.scale = alpha / rank

            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            original_out = self.original(x)
            lora_out = self.lora_B(self.lora_A(x))
            return original_out + lora_out * self.scale

    # Find and wrap target layers
    lora_layers = []
    target_names = []

    for name, module in cfm.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name for k in ["to_q", "to_k", "to_v", "to_out.0", "proj_out", "ff.net.2"]):
                lora_layer = LoRALayer(module).cuda()
                # Replace in parent
                parts = name.split(".")
                parent = cfm
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], lora_layer)
                lora_layers.append(lora_layer)
                target_names.append(name)

    print(f"\nApplied LoRA to {len(target_names)} layers:")
    for n in target_names:
        print(f"  - {n}")

    # Collect LoRA parameters only
    lora_params = []
    for layer in lora_layers:
        lora_params.extend(layer.lora_A.parameters())
        lora_params.extend(layer.lora_B.parameters())

    total_lora = sum(p.numel() for p in lora_params)
    print(f"\nLoRA params: {total_lora:,} ({total_lora/total_params*100:.2f}% of base)")

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
    print("\nStarting LoRA training (2000 steps, LR 1e-5)...")
    optimizer = torch.optim.AdamW(lora_params, lr=1e-5, weight_decay=0.01)
    batch_size = 2
    save_dir = "/ckpt/indicf5_kannada_lora_v4"
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
    for step in range(2000):
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
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            avg = np.mean(losses[-10:]) if losses else loss.item()
            print(f"  Step {step:4d}: loss={loss.item():.4f}, avg={avg:.4f}")

        # Checkpoint every 200 steps
        if (step + 1) % 200 == 0:
            ckpt_path = f"{save_dir}/step_{step+1:04d}"
            os.makedirs(ckpt_path, exist_ok=True)
            lora_state = {}
            for i, layer in enumerate(lora_layers):
                lora_state[f"lora_A_{i}"] = layer.lora_A.cpu().state_dict()
                lora_state[f"lora_B_{i}"] = layer.lora_B.cpu().state_dict()
                lora_state[f"scale_{i}"] = layer.scale
            lora_state["target_names"] = target_names
            lora_state["rank"] = RANK
            lora_state["alpha"] = ALPHA
            torch.save(lora_state, f"{ckpt_path}/lora.pt")
            with open(f"{ckpt_path}/config.json", "w") as f:
                json.dump({
                    "step": step+1,
                    "loss": float(np.mean(losses[-50:])),
                    "rank": RANK,
                    "alpha": ALPHA,
                    "lr": 1e-5,
                    "num_lora_layers": len(target_names),
                    "total_lora_params": total_lora,
                }, f)
            print(f"  ✓ Saved {ckpt_path} (avg loss: {np.mean(losses[-50:]):.4f})")
            data_vol.commit()

    # Final save
    final_dir = f"{save_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    lora_state = {}
    for i, layer in enumerate(lora_layers):
        lora_state[f"lora_A_{i}"] = layer.lora_A.cpu().state_dict()
        lora_state[f"lora_B_{i}"] = layer.lora_B.cpu().state_dict()
        lora_state[f"scale_{i}"] = layer.scale
    lora_state["target_names"] = target_names
    lora_state["rank"] = RANK
    lora_state["alpha"] = ALPHA
    torch.save(lora_state, f"{final_dir}/lora.pt")
    data_vol.commit()

    print(f"\n{'='*60}")
    print(f"✅ DONE — Final loss: {np.mean(losses[-50:]):.4f}")
    print(f"Saved to: {save_dir}")
    print(f"LoRA layers: {len(target_names)}, Rank: {RANK}")


if __name__ == "__main__":
    modal.run(train)
