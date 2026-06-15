"""Pipeline test: translate + Kannada TTS with user's reference voice.

Skips VoxCPM2 (already verified). Tests IndicTrans2 → IndicF5 with v2_step0500.
"""

import modal

app = modal.App("kannada-pipeline-test")
hf_cache = modal.Volume.from_name("dreamvoice-hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("dreamvoice-ckpt", create_if_missing=True)
data_vol = modal.Volume.from_name("dreamvoice-ft-data", create_if_missing=True)

INDICTRANS_ID = "ai4bharat/indictrans2-en-indic-1B"
INDICF5_ID = "ai4bharat/IndicF5"
REF_SENTENCE = "Once upon a time, in a cozy little house, a gentle voice began to tell a bedtime story."

trans_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==2.11.0", "transformers==4.51.3", "sentencepiece==0.2.0")
    .pip_install("git+https://github.com/VarunGumma/IndicTransToolkit.git")
)

f5_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.3",
        "soundfile==0.14.0", "numpy==1.26.4", "huggingface_hub",
    )
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
)

STORY_TEXT = "A brave little fox lived in a magical forest. Every night, the moon would sing a lullaby to help the animals fall asleep. One evening, the fox found a tiny star that had fallen from the sky. The fox gently carried the star back to the moon, who smiled and thanked the little fox."


@app.function(image=trans_image, gpu="A10G", timeout=600,
              volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("dreamvoice-secrets")])
def translate_story(english_text: str) -> str:
    import os, torch
    token = os.environ.get("HF_TOKEN") or None
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    tokenizer = AutoTokenizer.from_pretrained(INDICTRANS_ID, trust_remote_code=True, cache_dir="/cache", token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(INDICTRANS_ID, trust_remote_code=True, cache_dir="/cache", token=token)
    model = model.cuda()
    print("  ✓ IndicTrans2 loaded")

    ip = IndicProcessor(inference=True)
    sents = [english_text[i:i+180] for i in range(0, len(english_text), 180)]
    batch = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_length=512, num_beams=5)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang="kan_Knda")
    kannada = " ".join(t.strip() for t in translations if t.strip())
    print(f"  ✓ Kannada: {kannada}")
    return kannada


@app.function(image=f5_image, gpu="A10G", timeout=900,
              volumes={"/cache": hf_cache, "/ckpt": ckpt_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("dreamvoice-secrets")])
def tts_kannada(kannada_text: str):
    import os, torch, soundfile as sf, numpy as np, time
    token = os.environ.get("HF_TOKEN") or None

    # Load model
    from transformers import AutoModel
    model = AutoModel.from_pretrained(INDICF5_ID, trust_remote_code=True, cache_dir="/cache", token=token)
    for cfm_path in [
        "/ckpt/indicf5_kannada_v2/step_0500/cfm.pt",
        "/ckpt/indicf5_kannada_v3/step_0400/cfm.pt",
    ]:
        if os.path.exists(cfm_path):
            cfm_state = torch.load(cfm_path, map_location="cpu", weights_only=True)
            model.ema_model.load_state_dict(cfm_state)
            print(f"  ✓ Loaded {cfm_path}")
            break
    model = model.cuda()
    print("  ✓ IndicF5 on GPU")

    # Load user reference voice
    ref_path = "/data/reference_voice.wav"
    if not os.path.exists(ref_path):
        ref_path = "/data/user_ref/resampled_24k.wav"
    print(f"  ✓ Reference: {ref_path}")

    # Generate per sentence
    t0 = time.time()
    sentences = [s.strip() for s in kannada_text.split("।") if s.strip()]
    all_audio = []
    for sent in sentences:
        with torch.no_grad():
            audio_out = model(sent, ref_audio_path=ref_path, ref_text=REF_SENTENCE)
        audio = np.asarray(audio_out, dtype=np.float32).flatten()
        if audio.max() > 1.0:
            audio = audio / 32768.0
        all_audio.append(audio)
        print(f"    → {sent[:40]}... ({len(audio)/24000:.1f}s)")
    elapsed = time.time() - t0

    # Concat with pauses
    pause = np.zeros(int(24000 * 0.45), dtype=np.float32)
    final = []
    for i, a in enumerate(all_audio):
        if i > 0:
            final.append(pause)
        final.append(a)
    full_audio = np.concatenate(final) if final else np.zeros(24000, dtype=np.float32)

    out_path = "/data/pipeline_test_kannada.wav"
    sf.write(out_path, full_audio, 24000)
    rms = float(np.sqrt(np.mean(full_audio**2)))
    peak = float(np.max(np.abs(full_audio)))
    dur = len(full_audio) / 24000
    silence_pct = float(np.mean(np.abs(full_audio) < 10**(-35.0/20.0))) * 100

    print(f"\n  ✓ Total: {dur:.1f}s, gen_time {elapsed:.1f}s, RMS {rms:.4f}, Peak {peak:.4f}, Silence {silence_pct:.1f}%")
    print(f"  ✓ Saved: {out_path}")


@app.local_entrypoint()
def main():
    import time as _t
    t0 = _t.time()
    print("\n🎬 PIPELINE TEST: Translate + Kannada TTS\n" + "=" * 50)

    kannada = translate_story.remote(STORY_TEXT)
    t1 = _t.time()
    print(f"\n  ⏱ Translate: {t1-t0:.1f}s")

    tts_kannada.remote(kannada)
    t2 = _t.time()
    print(f"  ⏱ TTS: {t2-t1:.1f}s")
    print(f"\n  ⏱ Total: {t2-t0:.1f}s")
    print("\n✅ DONE")


if __name__ == "__main__":
    modal.run(main)
