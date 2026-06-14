"""Publish IndicF5 v2 Kannada checkpoint to HuggingFace.

Pushes the fine-tuned CFM weights to mitvho09/IndicF5-Kannada-Bedtime-v2.
"""

import modal

app = modal.App("indicf5-publish-v2")
data_vol = modal.Volume.from_name("dreamvoice-ft-data")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.4.1", "transformers==4.46.3",
        "safetensors", "huggingface_hub",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)


@app.function(
    image=image, gpu="A100-80GB", timeout=1800,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def publish():
    import os
    import torch
    from huggingface_hub import HfApi

    print("=" * 60)
    print("Publishing IndicF5 v2 Kannada to HuggingFace")
    print("=" * 60)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Find the best checkpoint (step_0500)
    ckpt_dir = "/ckpt/indicf5_kannada_v2/step_0500"
    if not os.path.exists(ckpt_dir):
        ckpt_dir = "/ckpt/indicf5_kannada_v2/final"
    if not os.path.exists(ckpt_dir):
        # List available
        available = os.listdir("/ckpt/indicf5_kannada_v2/")
        print(f"Available checkpoints: {available}")
        # Use the last one
        step_dirs = sorted([d for d in available if d.startswith("step_")])
        if step_dirs:
            ckpt_dir = f"/ckpt/indicf5_kannada_v2/{step_dirs[-1]}"
        else:
            ckpt_dir = "/ckpt/indicf5_kannada_v2/final"

    print(f"Using checkpoint: {ckpt_dir}")

    # Load the model and apply fine-tuned weights
    from transformers import AutoModel
    print("Loading base IndicF5...")
    model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5", trust_remote_code=True, cache_dir="/data/cache", token=token
    )

    # Load fine-tuned CFM weights
    cfm_path = f"{ckpt_dir}/cfm.pt"
    if os.path.exists(cfm_path):
        print(f"Loading fine-tuned weights from {cfm_path}")
        cfm_state = torch.load(cfm_path, map_location="cpu")
        model.ema_model.load_state_dict(cfm_state)
        print("✓ Weights loaded")
    else:
        print(f"✗ No cfm.pt found in {ckpt_dir}")
        return

    # Save locally
    local_dir = "/data/indicf5_kannada_v2_model"
    os.makedirs(local_dir, exist_ok=True)

    # Save as safetensors
    print("Saving model...")
    model.save_pretrained(local_dir)
    print(f"✓ Saved to {local_dir}")

    # Upload to HuggingFace
    repo_id = "mitvho09/IndicF5-Kannada-Bedtime-v2"
    print(f"\nUploading to {repo_id}...")

    try:
        api.create_repo(repo_id, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ Published: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Trying with git...")
        import subprocess
        subprocess.run(["git", "config", "--global", "user.email", "mitvho09@users.noreply.huggingface.co"])
        subprocess.run(["git", "config", "--global", "user.name", "mitvho09"])
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ Published via git: https://huggingface.co/{repo_id}")

    return {"status": "published", "repo": repo_id}


if __name__ == "__main__":
    modal.run(publish)
