# KANNADA_EXPERIMENT.md — Honest path to Kannada narration

> This is a **separate, parallel experiment** run by the human (you) — NOT a Codex app task and
> NOT on the critical path. Keep it out of the core submission claims until it actually sounds good.

## The honest situation
- **VoxCPM2 supports 30 languages, but Kannada is NOT one of them.** Its only Indic language is **Hindi**.
- **Cohere Transcribe** (14 languages) has no Kannada. **Tiny Aya** covers many Indic languages but
  Kannada support is unconfirmed.
- Therefore "mom's voice narrating a Kannada story" is **not** a turnkey, verifiable claim.
  The core app ships **English + Hindi** (both native, fully verifiable). Judges can verify those.

## Why still attempt it
- It's your mother tongue — personally meaningful and a genuinely unique demo.
- Fine-tuning VoxCPM2 and publishing the adapter on HF earns the **"Fine-tuned model" merit badge**.
- If it sounds good, it becomes a headline emotional moment. If not, it stays a labeled **beta**
  and never undermines the verified core.

## Recipe (run on a Modal GPU while Codex builds the app)
1. **Collect data.** Record the parent reading Kannada — for a real new-language adaptation aim for
   **500+ clean clips** (full fine-tune territory); a LoRA can start from far less but new-language
   LoRA is known to be finicky. Each clip: clean audio + accurate Kannada transcript.
2. **Start with LoRA**, then escalate to full SFT if quality is poor:
   ```bash
   python scripts/train_voxcpm_finetune.py \
     --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
   ```
   Suggested config for language adaptation: **LoRA rank r = 32–64**, **alpha = r or 2·r**,
   **`enable_dit: true`**, and **increase the stop-loss weight** (the stop loss and diffusion loss
   converge at different rates for a new language). See the VoxCPM2 fine-tuning docs.
3. **Expect failure modes.** A known issue is "garbage output" from rushed new-language LoRA. Budget
   for several attempts; if LoRA can't get there, full fine-tune with 500+ clips is the documented path.
4. **Evaluate honestly.** Have a Kannada speaker rate intelligibility. Ship to the app **only** if it's
   clearly understandable.
5. **Publish** the adapter/checkpoint to Hugging Face → claim the **Fine-tuned model badge**.
6. **Wire into the app** only via `ENABLE_KANNADA_BETA` (Codex task **T3.5**), clearly labeled
   "experimental (beta)".

## Fallback if fine-tuning doesn't land in time
- Demo **Hindi** as the Indian-language story (native, high quality), and list Kannada as
  "experimental / coming soon." This is fully honest and still resonates with an Indian audience.
- Optional weak fallback: generate the Kannada story, transliterate to Latin phonetics, feed to
  VoxCPM2. Quality is unreliable — only as a clearly-labeled curiosity, never a headline claim.

## The rule
**Never put an unverifiable Kannada claim in the pitch, README headline, or demo narration.**
Verified: English, Hindi. Experimental: Kannada (only if it genuinely works).
