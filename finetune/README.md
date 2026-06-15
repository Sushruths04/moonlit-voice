# DreamVoice fine-tuning — for the "Fine-tuned model" merit badge

The badge just requires **publishing a fine-tuned model on Hugging Face**. We do that with the
**lowest-risk, best-documented** path, and keep the riskier Kannada one optional.

## Two paths

### Path A — RECOMMENDED (low risk, documented): VoxCPM2 speaker LoRA
Fine-tune **VoxCPM2** on **5–10 minutes of the parent's voice** so English narration is more
consistent and personal. VoxCPM2 has a documented LoRA pipeline (JSONL manifest + YAML config +
`train_voxcpm_finetune.py`). This reliably produces a publishable adapter → **badge**, and improves
the core English demo. Run it on Modal GPU; it trains while you build the rest.

### Path B — OPTIONAL (higher risk): IndicF5 on Kannada children's speech
Stock IndicF5 already speaks Kannada well (no fine-tune needed for the feature). Only fine-tune if
you want *more expressive, kid-energetic* Kannada and have time. Data: OpenSLR **SLR79**
(Kannada, 347h / 915 speakers, https://openslr.org/79/) — take ~15–30 min of one clear, lively
speaker. IndicF5 is F5-TTS-based; confirm its training entrypoint from the AI4Bharat/IndicF5 repo
before launching (marked `TODO(verify)` in `run_modal.py`).

## Steps (Path A)
1. Put the parent's clips in `finetune/data/raw/` (WAV/MP3, 3–15 s each, clean) and a
   `finetune/data/transcripts.tsv` with `filename<TAB>exact transcript` per line.
2. `python finetune/prepare_data.py` → validates, resamples to 16 kHz, writes
   `data/manifest_train.jsonl` + `data/manifest_val.jsonl`. Fails loud if data is bad.
3. Edit `voxcpm_finetune_lora.yaml` if needed (paths, steps).
4. `modal run finetune/run_modal.py` → trains with **checkpoint + resume + auto-eval** on a Modal
   Volume. Re-running resumes from the last checkpoint (never restarts from zero).
5. The job publishes the best adapter to your HF repo (set `HF_REPO`, `HF_TOKEN`). Link it in the
   app README → **badge claimed**.

## Failproof / self-improving properties
- Resumable (checkpoints to a Modal Volume; rerun continues).
- Self-checking (asserts manifest non-empty, audio readable, eval clip produced; fails loud).
- Auto-eval each saved checkpoint on a fixed sentence; keeps the best by eval loss.
- No secrets in code (HF/Modal creds from env). Everything headless → OpenCode can run + verify.
