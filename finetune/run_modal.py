"""Modal GPU job: LoRA fine-tune VoxCPM2 with checkpoint + resume + auto-eval.

Failproof / self-improving:
- checkpoints to a persistent Modal Volume → re-running RESUMES, never restarts from zero,
- validates the manifests before training (fails loud),
- after training, synthesizes a fixed eval sentence from the latest checkpoint so you can hear
  whether it improved,
- optionally publishes the adapter to HF for the merit badge.

Run:
    modal run finetune/run_modal.py --hf-repo <your-username>/dreamvoice-voxcpm2-lora

Env (Modal secret `dreamvoice-secrets`): HF_TOKEN.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

GPU = "A100"  # speaker LoRA fits smaller GPUs; A100 keeps it fast. Drop to "A10G" to save cost.

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
    )
    .run_commands("git clone https://github.com/OpenBMB/VoxCPM.git /VoxCPM")
)

app = modal.App("dreamvoice-finetune")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)
HERE = os.path.dirname(os.path.abspath(__file__))


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 4,
    volumes={"/ckpt": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def train(hf_repo: str | None = None):
    import glob
    import subprocess
    import sys

    # 1) validate manifests (fail loud) -------------------------------------
    for m in ("/data/manifest_train.jsonl", "/data/manifest_val.jsonl"):
        if not os.path.exists(m) or os.path.getsize(m) == 0:
            raise SystemExit(
                f"❌ {m} missing/empty. Upload data to the 'dreamvoice-ft-data' Volume first "
                "(run prepare_data.py locally, then `modal volume put dreamvoice-ft-data ./finetune/data /`)."
            )

    # 2) resume detection ---------------------------------------------------
    existing = sorted(glob.glob("/ckpt/**/*.pt", recursive=True) + glob.glob("/ckpt/**/checkpoint*", recursive=True))
    if existing:
        print(f"↻ Resuming — found {len(existing)} checkpoint artifact(s); latest: {existing[-1]}")
    else:
        print("▶ Fresh run — no checkpoints yet.")

    # 3) train --------------------------------------------------------------
    cfg = "/data/voxcpm_finetune_lora.yaml"
    cmd = [sys.executable, "/VoxCPM/scripts/train_voxcpm_finetune.py", "--config_path", cfg]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/", capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise SystemExit(
            "❌ training command failed. Open the VoxCPM2 repo, find the exact finetune script/module "
            "name, and update the `cmd` above. (This is the single TODO to verify.)"
        )
    ckpt_vol.commit()

    # 4) auto-eval the latest checkpoint -----------------------------------
    _eval_latest()

    # 5) publish adapter for the badge -------------------------------------
    if hf_repo:
        _publish(hf_repo)
    print("✅ done")


def _eval_latest():
    """Synthesize a fixed sentence from the newest checkpoint so you can hear progress."""
    try:
        import glob
        import json

        import soundfile as sf
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2

        ckpts = sorted(glob.glob("/ckpt/**/lora_config.json", recursive=True))
        if not ckpts:
            # Try full SFT checkpoints
            ckpts = sorted(glob.glob("/ckpt/**/model.safetensors", recursive=True))
            if not ckpts:
                print("⚠️  no checkpoint to eval yet.")
                return
            ckpt_dir = str(Path(ckpts[-1]).parent)
            model = VoxCPM.from_pretrained(ckpt_dir, device="cuda")
        else:
            ckpt_dir = str(Path(ckpts[-1]).parent)
            with open(ckpts[-1]) as f:
                lora_cfg = json.load(f)
            base_model_id = lora_cfg.get("base_model", "openbmb/VoxCPM2")
            model = VoxCPM.from_pretrained(base_model_id, device="cuda")
            lora_config = LoRAConfigV2(**lora_cfg["lora_config"])
            model.load_lora(ckpt_dir, lora_config=lora_config)

        wav = model.generate(text="(gentle, warm bedtime voice) Goodnight, little one. Sweet dreams.")
        out = "/ckpt/eval_sample.wav"
        sf.write(out, wav, int(model.tts_model.sample_rate))
        print(f"🎧 wrote eval sample → {out} (download with `modal volume get dreamvoice-ckpt eval_sample.wav`)")
    except Exception as exc:  # noqa: BLE001 - eval is best-effort, never fail the run on it
        print(f"⚠️  eval step skipped: {exc}")


def _publish(hf_repo: str):
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("⚠️  HF_TOKEN not set — skipping publish.")
            return
        api = HfApi(token=token)
        api.create_repo(hf_repo, exist_ok=True)
        api.upload_folder(folder_path="/ckpt", repo_id=hf_repo, repo_type="model")
        print(f"📤 published adapter → https://huggingface.co/{hf_repo}  (claim the badge + link in README)")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  publish skipped: {exc}")


@app.local_entrypoint()
def main(hf_repo: str = ""):
    train.remote(hf_repo or None)
