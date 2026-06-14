"""Compare stock VoxCPM2 vs best LoRA checkpoint (exp6) on the same sentences."""

import os
import modal

GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "voxcpm==2.0.3",
        "transformers==5.12.0",
        "soundfile==0.14.0",
        "numpy==2.2.6",
    )
)

app = modal.App("dreamvoice-compare")
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt")
data_vol = modal.Volume.from_name("dreamvoice-ft-data")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={"/ckpt": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def compare():
    import soundfile as sf
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2

    sentences = [
        "(gentle, warm bedtime voice) Once upon a time, in a cozy little house, a gentle voice began to tell a bedtime story.",
        "(soft, dreamy whisper) The little fox curled up by the fire as snow fell softly outside.",
        "(calm, slow pace) Goodnight, little one. Sweet dreams among the stars.",
        "(warm, loving tone) The moon smiled down as the children closed their eyes, safe and warm.",
    ]

    # Load stock model
    print("Loading stock VoxCPM2...")
    stock_model = VoxCPM.from_pretrained("/data/voxcpm2_model", device="cuda")

    # Load best LoRA checkpoint (exp6)
    print("Loading LoRA checkpoint (exp6)...")
    import json, glob
    ckpts = sorted(glob.glob("/ckpt/exp6_r32_lr5e5_half/**/lora_config.json", recursive=True))
    if ckpts:
        with open(ckpts[-1]) as f:
            lora_cfg = json.load(f)
        lora_config = LoRAConfigV2(**lora_cfg["lora_config"])
        lora_model = VoxCPM.from_pretrained(
            "/data/voxcpm2_model", device="cuda", lora_config=lora_config
        )
        lora_model.load_lora(str(os.path.dirname(ckpts[-1])))
    else:
        print("No LoRA checkpoint found!")
        return

    os.makedirs("/data/comparison", exist_ok=True)

    for i, sentence in enumerate(sentences):
        print(f"Sentence {i+1}: {sentence[:60]}...")

        # Stock
        wav_stock = stock_model.generate(text=sentence)
        sf.write(f"/data/comparison/stock_{i}.wav", wav_stock, int(stock_model.tts_model.sample_rate))

        # LoRA
        wav_lora = lora_model.generate(text=sentence)
        sf.write(f"/data/comparison/lora_exp6_{i}.wav", wav_lora, int(lora_model.tts_model.sample_rate))

        print(f"  stock: {len(wav_stock)/stock_model.tts_model.sample_rate:.1f}s")
        print(f"  lora:  {len(wav_lora)/lora_model.tts_model.sample_rate:.1f}s")

    # Also save the LoRA adapter for publishing
    import shutil
    lora_dir = str(os.path.dirname(ckpts[-1]))
    shutil.copytree(lora_dir, "/data/comparison/lora_adapter", dirs_exist_ok=True)

    data_vol.commit()
    print("\nDone! Files in /data/comparison/")
    print("Download with: modal volume get dreamvoice-ft-data comparison/ ./comparison/")


@app.local_entrypoint()
def main():
    compare.remote()
