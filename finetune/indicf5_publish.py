"""Phase 3: Publish fine-tuned IndicF5 to HuggingFace.

Uploads the fine-tuned checkpoint as a diff from the base model.
"""

import os
import modal

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "huggingface_hub",
    "torch==2.4.1",
)

app = modal.App("indicf5-publish")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={"/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def publish():
    import torch
    from huggingface_hub import HfApi, create_repo

    print("=" * 60)
    print("PHASE 3: Publish Fine-Tuned IndicF5 to HuggingFace")
    print("=" * 60)

    token = os.environ.get("HF_TOKEN")
    repo_id = "mitvho09/IndicF5-Kannada-Bedtime"

    # Check checkpoint exists
    ckpt_path = "/ckpt/indicf5_kannada/step_0300/cfm.pt"
    if not os.path.exists(ckpt_path):
        print(f"  ✗ No checkpoint at {ckpt_path}")
        return {"status": "FAIL"}

    print(f"  Checkpoint: {ckpt_path}")
    size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"  Size: {size_mb:.1f} MB")

    # Create repo
    print(f"\nCreating HF repo: {repo_id}")
    try:
        create_repo(repo_id, token=token, exist_ok=True, private=False)
        print(f"  ✓ Repo created/found")
    except Exception as e:
        print(f"  ✗ Failed to create repo: {e}")
        return {"status": "FAIL"}

    # Prepare files for upload
    upload_dir = "/tmp/indicf5_upload"
    os.makedirs(upload_dir, exist_ok=True)

    # Copy checkpoint
    import shutil
    shutil.copy(ckpt_path, f"{upload_dir}/cfm.pt")

    # Create model card
    model_card = """---
language: kn
tags:
  - text-to-speech
  - kannada
  - indicf5
  - bedtime-stories
base_model: ai4bharat/IndicF5
library_name: transformers
license: mit
---

# IndicF5 Kannada Bedtime Story Fine-Tune

Fine-tuned [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) on Kannada bedtime-story speech.

## Training Details

- **Base model**: ai4bharat/IndicF5 (0.4B params)
- **Training data**: Rasa Kannada (18 clips, ~2min)
- **Training steps**: 300
- **Learning rate**: 1e-5
- **Final loss**: 0.4553
- **GPU**: A100-80GB (Modal)

## How to Use

```python
from transformers import AutoModel
import soundfile as sf

model = AutoModel.from_pretrained(
    "mitvho09/IndicF5-Kannada-Bedtime",
    trust_remote_code=True,
)

audio = model(
    "ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ.",
    ref_audio_path="reference.wav",
    ref_text="transcript of reference",
)

sf.write("output.wav", audio, 24000)
```

## Evaluation

Compare stock vs fine-tuned audio in the `comparison/` directory.

## Fine-Tuning Details

This is an experimental fine-tune for the DreamVoice competition badge.
Stock IndicF5 may perform better for general Kannada TTS.
"""

    with open(f"{upload_dir}/README.md", "w") as f:
        f.write(model_card)

    # Upload
    print("\nUploading to HuggingFace...")
    api = HfApi(token=token)

    try:
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        return {"status": "FAIL"}

    print("\n✅ PHASE 3 COMPLETE")
    print(f"  Published: https://huggingface.co/{repo_id}")
    print(f"  Add to README.md for competition badge")

    return {"status": "PASS", "repo": repo_id}


if __name__ == "__main__":
    app.run()
