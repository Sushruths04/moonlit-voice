"""Generate stock vs v2 fine-tuned Kannada comparison samples.

Downloads WAVs locally for analysis.
"""

import modal
import io

app = modal.App("kannada-compare-live")
hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)

f5_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4",
        "safetensors", "huggingface_hub",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt", create_if_missing=True)

TEST_SENTENCES = [
    "ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ.",
    "ಒಂದು ಕಾಡಿನಲ್ಲಿ ಒಂದು ಚಿಕ್ಕ ಮರಿ ಹುಲಿ ವಾಸಿಸುತ್ತಿತ್ತು.",
    "ರಾತ್ರಿ ಬಂದಾಗ ಆಕಾಶದಲ್ಲಿ ನಕ್ಷತ್ರಗಳು ಮಿನುಗುತ್ತವೆ.",
]


@app.function(
    image=f5_image, gpu="A10G", timeout=600,
    volumes={"/cache": hf_cache, "/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("dreamvoice-secrets")],
)
def compare():
    import os, json, time
    import torch
    import soundfile as sf
    import numpy as np

    print("=" * 60)
    print("STOCK vs v2 FINE-TUNED COMPARISON")
    print("=" * 60)

    token = os.environ.get("HF_TOKEN") or None
    INDICF5_ID = "ai4bharat/IndicF5"
    V2_REPO = "mitvho09/IndicF5-Kannada-Bedtime-v2"

    from transformers import AutoModel

    # ─── Load reference ─────────────────────────────────────────
    ref_path = None
    ref_text = ""
    manifest_path = "/data/kannada_tts/manifest.jsonl"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for item in lines[:50]:
            try:
                p = item["audio"]
                if os.path.exists(p):
                    ref_path = p
                    ref_text = item["text"]
                    break
            except:
                continue
    if ref_path is None:
        print("ERROR: no reference audio found")
        return

    print(f"Reference: {ref_text[:60]}...")

    # ─── Generate stock ─────────────────────────────────────────
    print("\n--- STOCK IndicF5 ---")
    model = AutoModel.from_pretrained(
        INDICF5_ID, trust_remote_code=True, cache_dir="/cache", token=token)
    try:
        model = model.to("cuda")
    except:
        pass

    os.makedirs("/data/compare_stock", exist_ok=True)
    stock_results = []
    for i, text in enumerate(TEST_SENTENCES):
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model(text, ref_audio_path=ref_path, ref_text=ref_text)
            dt = time.time() - t0
            audio = np.asarray(out, dtype=np.float32).flatten()
            if audio.max() > 1.0:
                audio = audio / 32768.0
            dur = len(audio) / 24000
            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            sil = float(np.mean(np.abs(audio) < 10**(-35.0/20.0)) * 100)
            sf.write(f"/data/compare_stock/stock_{i}.wav", audio, 24000)
            stock_results.append({"i":i,"dur":round(dur,2),"gen":round(dt,2),"rms":round(rms,4),"peak":round(peak,4),"sil":round(sil,1)})
            print(f"  [{i}] {dur:.1f}s gen={dt:.1f}s rms={rms:.4f} peak={peak:.4f} sil={sil:.1f}%")
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")

    # ─── Load v2 weights ────────────────────────────────────────
    print("\n--- v3 FINE-TUNED (step_0400, best loss) ---")
    try:
        import torch as _torch
        cfm_path = "/ckpt/indicf5_kannada_v3/step_0400/cfm.pt"
        if not os.path.exists(cfm_path):
            cfm_path = "/ckpt/indicf5_kannada_v3/final/cfm.pt"
        if os.path.exists(cfm_path):
            cfm_state = _torch.load(cfm_path, map_location="cpu", weights_only=True)
            model.ema_model.load_state_dict(cfm_state)
            print(f"  ✓ Loaded CFM from {cfm_path}")
        else:
            print(f"  ✗ No checkpoint found. Available: {os.listdir('/ckpt/indicf5_kannada_v3/')}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return

    os.makedirs("/data/compare_v2", exist_ok=True)
    v2_results = []
    for i, text in enumerate(TEST_SENTENCES):
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model(text, ref_audio_path=ref_path, ref_text=ref_text)
            dt = time.time() - t0
            audio = np.asarray(out, dtype=np.float32).flatten()
            if audio.max() > 1.0:
                audio = audio / 32768.0
            dur = len(audio) / 24000
            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            sil = float(np.mean(np.abs(audio) < 10**(-35.0/20.0)) * 100)
            sf.write(f"/data/compare_v2/v2_{i}.wav", audio, 24000)
            v2_results.append({"i":i,"dur":round(dur,2),"gen":round(dt,2),"rms":round(rms,4),"peak":round(peak,4),"sil":round(sil,1)})
            print(f"  [{i}] {dur:.1f}s gen={dt:.1f}s rms={rms:.4f} peak={peak:.4f} sil={sil:.1f}%")
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    if stock_results and v2_results:
        for metric in ["dur","gen","rms","peak","sil"]:
            s = np.mean([r[metric] for r in stock_results])
            v = np.mean([r[metric] for r in v2_results])
            label = {"dur":"Duration(s)","gen":"GenTime(s)","rms":"RMS Energy","peak":"Peak","sil":"Silence(%)"}[metric]
            diff = ((v - s) / s * 100) if s != 0 else 0
            print(f"  {label:15s}  Stock: {s:.3f}  v2: {v:.3f}  ({diff:+.1f}%)")

    data_vol.commit()
    print("\n✅ DONE — Files in /data/compare_stock/ and /data/compare_v2/")


if __name__ == "__main__":
    modal.run(compare)
