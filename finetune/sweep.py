"""Modal GPU sweep: run 6 VoxCPM2 LoRA experiments in parallel.

Each experiment uses different hyperparameters (rank, LR, data size).
All checkpoints go to separate Modal Volumes so we can compare.

Run:
    modal run finetune/sweep.py

Env (Modal secret `dreamvoice-secrets`): HF_TOKEN.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

GPU = "A100-80GB"  # VoxCPM2 is 2B params; needs 80GB for LoRA training

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "voxcpm==2.0.3",
        "transformers==5.12.0",
        "accelerate==1.14.0",
        "soundfile==0.14.0",
        "librosa==0.11.0",
        "numpy==2.2.6",
        "huggingface_hub==1.19.0",
        "pyyaml==6.0.2",
        "datasets",
    )
    .run_commands("git clone https://github.com/OpenBMB/VoxCPM.git /VoxCPM")
)

app = modal.App("dreamvoice-sweep")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)

# Experiment configs: (name, lora_rank, lora_alpha, learning_rate, max_steps, data_fraction)
EXPERIMENTS = [
    ("exp1_r32_lr1e4",      32, 64,  1.0e-4, 100, 1.0),   # baseline
    ("exp2_r16_lr5e5",      16, 32,  5.0e-5, 100, 1.0),   # smaller rank, lower LR
    ("exp3_r64_lr1e4",      64, 128, 1.0e-4, 100, 1.0),   # larger rank
    ("exp4_r32_lr5e5",      32, 64,  5.0e-5, 200, 1.0),   # longer training, lower LR
    ("exp5_r32_lr2e4",      32, 64,  2.0e-4, 100, 1.0),   # higher LR
    ("exp6_r32_lr5e5_half", 32, 64,  5.0e-5, 100, 0.5),   # half data
]


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 3,  # 3 hours per experiment
    volumes={"/ckpt": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def run_experiment(name: str, lora_rank: int, lora_alpha: int, lr: float, max_steps: int, data_fraction: float):
    import json
    import subprocess
    import sys

    print(f"\n{'='*60}")
    print(f"🚀 Starting {name}")
    print(f"   rank={lora_rank}, alpha={lora_alpha}, lr={lr}, steps={max_steps}, data={data_fraction:.0%}")
    print(f"{'='*60}\n")

    # 0) Extract data + download model if needed
    if not os.path.exists("/data/clean"):
        print("📦 Extracting training data...")
        subprocess.run(["tar", "xzf", "/data/subset.tar.gz", "-C", "/data/"], check=True)
        # Rename subset/ to clean/
        if os.path.exists("/data/subset"):
            os.rename("/data/subset", "/data/clean")
        print("   Done.")

    model_dir = "/data/voxcpm2_model"
    if not os.path.exists(f"{model_dir}/config.json"):
        print("📦 Downloading VoxCPM2 model...")
        from huggingface_hub import snapshot_download
        snapshot_download("openbmb/VoxCPM2", local_dir=model_dir, token=os.environ.get("HF_TOKEN"))
        print("   Done.")

    # 1) Validate data exists
    for m in ("/data/manifest_train.jsonl", "/data/manifest_val.jsonl"):
        if not os.path.exists(m) or os.path.getsize(m) == 0:
            raise SystemExit(
                f"❌ {m} missing/empty. Run download_storynory.py first, then upload to Modal Volume."
            )

    # 2) Optionally subsample data
    if data_fraction < 1.0:
        _subsample_data(data_fraction)

    # 3) Create experiment output dir
    ckpt_dir = f"/ckpt/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 4) Write experiment config
    cfg = {
        "pretrained_path": "/data/voxcpm2_model",
        "train_manifest": "/data/manifest_train.jsonl",
        "val_manifest": "/data/manifest_val.jsonl",
        "sample_rate": 16000,
        "out_sample_rate": 48000,
        "lora": {
            "r": lora_rank,
            "alpha": lora_alpha,
            "enable_dit": True,
        },
        "batch_size": 1,
        "grad_accum_steps": 16,
        "learning_rate": lr,
        "num_iters": max_steps,
        "max_steps": max_steps,
        "warmup_steps": min(100, max_steps // 5),
        "save_path": ckpt_dir,
        "save_interval": max(50, max_steps // 5),
        "valid_interval": max(50, max_steps // 5),
    }
    cfg_path = f"/data/config_{name}.yaml"
    # Write as YAML (not JSON) to avoid type parsing issues
    import yaml
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # 5) Train
    cmd = [sys.executable, "/VoxCPM/scripts/train_voxcpm_finetune.py", "--config_path", cfg_path]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/", capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        print(f"⚠️  {name} training failed — check logs above")
        return

    # 6) Auto-eval best checkpoint
    _eval_best(ckpt_dir, name)
    ckpt_vol.commit()

    print(f"\n✅ {name} complete!")


def _subsample_data(fraction: float):
    """Create subsampled manifests."""
    import random

    for split in ("train", "val"):
        src = f"/data/manifest_{split}.jsonl"
        dst = f"/data/manifest_{split}_sub.jsonl"
        with open(src) as f:
            lines = f.readlines()
        n = max(1, int(len(lines) * fraction))
        random.seed(42)
        sampled = random.sample(lines, n)
        with open(dst, "w") as f:
            f.writelines(sampled)
        # Point training to subsampled data
        os.system(f"cp {dst} {src}")


def _eval_best(ckpt_dir: str, exp_name: str):
    """Synthesize eval sentence from best checkpoint."""
    try:
        import glob
        import json

        import soundfile as sf
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2

        ckpts = sorted(glob.glob(f"{ckpt_dir}/**/lora_config.json", recursive=True))
        if not ckpts:
            print(f"⚠️  {exp_name}: no checkpoint to eval")
            return

        ckpt_path = ckpts[-1]
        ckpt_dir_path = str(Path(ckpt_path).parent)

        with open(ckpt_path) as f:
            lora_cfg = json.load(f)

        base_model_id = lora_cfg.get("base_model", "openbmb/VoxCPM2")
        model = VoxCPM.from_pretrained(base_model_id, device="cuda")
        lora_config = LoRAConfigV2(**lora_cfg["lora_config"])
        model.load_lora(ckpt_dir_path, lora_config=lora_config)

        eval_sentences = [
            "(gentle, warm bedtime voice) Once upon a time, in a cozy little house, a gentle voice began to tell a bedtime story.",
            "(soft, dreamy whisper) The little fox curled up by the fire as snow fell softly outside.",
            "(calm, slow pace) Goodnight, little one. Sweet dreams among the stars.",
        ]

        for j, sentence in enumerate(eval_sentences):
            wav = model.generate(text=sentence)
            out = f"{ckpt_dir_path}/eval_{j}.wav"
            sf.write(out, wav, int(model.tts_model.sample_rate))
            print(f"   🎧 {exp_name} eval {j}: {out}")

    except Exception as exc:
        print(f"⚠️  {exp_name} eval skipped: {exc}")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={"/ckpt": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def compare_checkpoints():
    """Load all experiment checkpoints, synthesize same sentence, save for comparison."""
    try:
        import glob
        import json

        import soundfile as sf
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2

        eval_sentence = "(gentle, warm bedtime voice) Once upon a time, in a cozy little house, a gentle voice began to tell a bedtime story. The stars twinkled gently outside the window."

        results = []
        for exp_dir in sorted(glob.glob("/ckpt/exp*")):
            exp_name = os.path.basename(exp_dir)
            ckpts = sorted(glob.glob(f"{exp_dir}/**/lora_config.json", recursive=True))
            if not ckpts:
                continue

            try:
                with open(ckpts[-1]) as f:
                    lora_cfg = json.load(f)
                base_model_id = lora_cfg.get("base_model", "openbmb/VoxCPM2")
                model = VoxCPM.from_pretrained(base_model_id, device="cuda")
                lora_config = LoRAConfigV2(**lora_cfg["lora_config"])
                model.load_lora(str(Path(ckpts[-1]).parent), lora_config=lora_config)

                wav = model.generate(text=eval_sentence)
                out = f"/data/comparison_{exp_name}.wav"
                sf.write(out, wav, int(model.tts_model.sample_rate))
                results.append((exp_name, out))
                print(f"✅ {exp_name} → {out}")

                del model
                import torch
                torch.cuda.empty_cache()

            except Exception as exc:
                print(f"⚠️  {exp_name} failed: {exc}")

        data_vol.commit()
        print(f"\n🎧 Comparison files saved to /data/comparison_*.wav")
        print("Download with: modal volume get dreamvoice-ft-data comparison_*.wav .")

    except Exception as exc:
        print(f"⚠️  comparison failed: {exc}")


@app.local_entrypoint()
def main():
    import concurrent.futures

    print("🚀 Starting training sweep — 6 experiments in parallel")
    print("   Each runs on A10G GPU for up to 3 hours")
    print()

    # Run all experiments
    for name, rank, alpha, lr, steps, data_frac in EXPERIMENTS:
        run_experiment.spawn(name, rank, alpha, lr, steps, data_frac)

    print("\n⏳ All experiments submitted. Check Modal dashboard for progress.")
    print("   After completion, run: modal run finetune/sweep.py --compare")


@app.local_entrypoint()
def compare():
    compare_checkpoints.remote()
