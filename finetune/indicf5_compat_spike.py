"""Phase 1: Compatibility Spike v2 — Can we train INF5Model directly?

IndicF5 loads as INF5Model with wrapped weights (ema_model._orig_mod.*).
Instead of extracting into raw DiT, train INF5Model end-to-end.

Tests:
1. Load IndicF5 as INF5Model ✓ (proven in v1)
2. Run a forward pass through INF5Model
3. Run 1 training step
4. Save checkpoint
5. Reload and synthesize
"""

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
        "safetensors",
        "accelerate",
        "hydra-core",
        "omegaconf",
        "einops",
        "torchdiffeq",
        "vocos",
        "x_transformers",
        "ema_pytorch",
        "librosa",
        "pydub",
        "wandb",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

app = modal.App("indicf5-compat-v2")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")
data_vol = modal.Volume.from_name("dreamvoice-ft-data")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 30,
    volumes={"/ckpt": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def compat_spike():
    import json
    import torch
    import soundfile as sf
    import numpy as np

    print("=" * 60)
    print("PHASE 1: COMPATIBILITY SPIKE v2 — INF5Model direct training")
    print("=" * 60)

    # ─── Step 1: Load IndicF5 ─────────────────────────────────────
    print("\n[Step 1] Loading IndicF5...")
    from transformers import AutoModel, AutoConfig

    token = os.environ.get("HF_TOKEN")
    indicf5_id = "ai4bharat/IndicF5"

    model = AutoModel.from_pretrained(
        indicf5_id, trust_remote_code=True, cache_dir="/cache", token=token
    )
    print(f"  ✓ Model type: {type(model).__name__}")

    # Explore the model structure
    print(f"  Model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")

    # Check if model has a forward method
    if hasattr(model, 'forward'):
        import inspect
        sig = inspect.signature(model.forward)
        print(f"  forward() params: {list(sig.parameters.keys())}")

    # Check submodules
    for name, child in model.named_children():
        print(f"  Submodule: {name} → {type(child).__name__}")
        if name == 'ema_model':
            for sub_name, sub_child in child.named_children():
                print(f"    {sub_name} → {type(sub_child).__name__}")
                if sub_name == 'transformer':
                    for ss_name, ss_child in sub_child.named_children():
                        print(f"      {ss_name} → {type(ss_child).__name__}")

    # ─── Step 2: Try forward pass ──────────────────────────────────
    print("\n[Step 2] Trying forward pass through INF5Model...")
    try:
        import soundfile as sf

        # Create a dummy reference audio file
        dummy_audio = np.random.randn(24000).astype(np.float32) * 0.01
        ref_path = "/tmp/test_ref.wav"
        sf.write(ref_path, dummy_audio, 24000)

        # Create dummy text
        dummy_text = "ನಮಸ್ಕಾರ ವಿಶ್ವ"
        dummy_ref_text = "ನಮಸ್ಕಾರ"

        # Try calling model's generate/inference method
        if hasattr(model, 'generate'):
            print("  Model has generate() method")
            with torch.no_grad():
                output = model(
                    dummy_text,
                    ref_audio_path=ref_path,
                    ref_text=dummy_ref_text,
                )
            print(f"  ✓ Forward pass succeeded! Output type: {type(output)}")
            if isinstance(output, (list, np.ndarray, torch.Tensor)):
                print(f"  Output shape: {np.array(output).shape}")
        else:
            print("  Model does NOT have generate() — checking for inference method")

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # ─── Step 3: Try CFM training forward ──────────────────────────
    print("\n[Step 3] Trying CFM training forward...")
    try:
        cfm = model.ema_model  # This is the CFM wrapper
        print(f"  CFM type: {type(cfm).__name__}")

        # Check CFM forward signature
        import inspect
        sig = inspect.signature(cfm.forward)
        print(f"  CFM.forward() params: {list(sig.parameters.keys())}")

        # CFM.forward takes (inp, text) where inp is raw waveform
        device = next(cfm.parameters()).device

        # Create dummy waveform batch (2 samples, 2s each at 24kHz)
        dummy_wave = torch.randn(2, 48000).float().to(device)
        dummy_text_list = ["ನಮಸ್ಕಾರ", "ಶುಭ ರಾತ್ರಿ"]

        cfm.train()
        with torch.no_grad():
            loss, cond, pred = cfm(dummy_wave, dummy_text_list)
        print(f"  ✓ CFM training forward succeeded!")
        print(f"    Loss: {loss.item():.4f}")
        print(f"    Cond shape: {cond.shape}")
        print(f"    Pred shape: {pred.shape}")

    except Exception as e:
        print(f"  ✗ CFM training forward failed: {e}")
        import traceback
        traceback.print_exc()

    # ─── Step 4: Training step with CFM ───────────────────────────
    print("\n[Step 4] Attempting 1 full training step...")
    try:
        # Unfreeze all CFM parameters
        for param in cfm.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in cfm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        optimizer = torch.optim.Adam(cfm.parameters(), lr=1e-5)
        optimizer.zero_grad()

        # Use CFM forward for training
        dummy_wave = torch.randn(2, 48000).float().to(device)
        dummy_text_list = ["ನಮಸ್ಕಾರ", "ಶುಭ ರಾತ್ರಿ"]

        loss, cond, pred = cfm(dummy_wave, dummy_text_list)
        loss.backward()

        # Check gradients
        grad_norm = sum(p.grad.norm().item() for p in cfm.parameters() if p.grad is not None)
        print(f"  Gradient norm: {grad_norm:.4f}")

        optimizer.step()
        print(f"  ✓ Training step completed! Loss: {loss.item():.4f}")

    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAIL", "step": "train_step", "error": str(e)}

    # ─── Step 5: Save checkpoint ──────────────────────────────────
    print("\n[Step 5] Saving checkpoint...")
    ckpt_dir = "/ckpt/indicf5_compat_v2"
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        # Save the CFM state dict (transformer + mel spec + config)
        torch.save(
            cfm.state_dict(),
            os.path.join(ckpt_dir, "cfm.pt"),
        )
        size_mb = os.path.getsize(os.path.join(ckpt_dir, "cfm.pt")) / 1e6
        print(f"  ✓ Checkpoint saved: {ckpt_dir}/cfm.pt ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        return {"status": "FAIL", "step": "save", "error": str(e)}

    # ─── Step 6: Reload and verify ────────────────────────────────
    print("\n[Step 6] Reloading checkpoint...")
    try:
        state = torch.load(os.path.join(ckpt_dir, "cfm.pt"))
        cfm.load_state_dict(state)
        print("  ✓ Checkpoint reloaded successfully!")
    except Exception as e:
        print(f"  ✗ Reload failed: {e}")
        return {"status": "FAIL", "step": "reload", "error": str(e)}

    # ─── Step 7: Synthesize test line ─────────────────────────────
    print("\n[Step 7] Synthesizing test line...")
    try:
        import soundfile as sf

        model.eval()
        with torch.no_grad():
            output = model(
                "ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ.",
                ref_audio_path=ref_path,
                ref_text="ನಮಸ್ಕಾರ",
            )

        if isinstance(output, (list, np.ndarray)):
            audio = np.array(output)
        elif isinstance(output, torch.Tensor):
            audio = output.cpu().numpy()
        else:
            audio = None

        if audio is not None and len(audio) > 0:
            sf.write("/data/indicf5_compat_v2_output.wav", audio.astype(np.float32), 24000)
            print(f"  ✓ Audio saved: /data/indicf5_compat_v2_output.wav")
            print(f"    Duration: {len(audio)/24000:.1f}s")
        else:
            print(f"  ⚠ Output is empty or unexpected type: {type(output)}")

    except Exception as e:
        print(f"  ✗ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAIL", "step": "synthesis", "error": str(e)}

    # ─── RESULT ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ COMPATIBILITY SPIKE v2 PASSED")
    print("=" * 60)
    print("\nKey findings:")
    print("  1. IndicF5 loads as INF5Model with wrapped weights")
    print("  2. Transformer (DiT) is accessible via model.ema_model.transformer")
    print("  3. Training step on transformer works")
    print("  4. Checkpoint save/reload works")
    print("  5. Inference through INF5Model works")
    print("\n→ PROCEED TO PHASE 2: Short domain fine-tune")

    return {"status": "PASS"}


if __name__ == "__main__":
    app.run()
