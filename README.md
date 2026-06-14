---
title: DreamVoice
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: true
tags:
  - thousand-token-wood
  - openbmb
  - minicpm
  - voxcpm2
  - voice-cloning
  - bedtime-stories
  - indicf5
  - indictrans2
---

# 🌙 DreamVoice — Bedtime stories in Mom's own voice

> Mom is traveling for work. Her 6-year-old can't sleep. She opens DreamVoice, taps
> **🐉 Dragons + 😄 Funny**, and hears *her mom's voice* — cloned, warm, familiar — tell a
> story about a clumsy dragon who sneezes glitter. She's asleep in three minutes, smiling.

DreamVoice clones a parent's voice from a short recording and narrates a custom, AI-generated
bedtime story in it — so a child hears a loved one read to them, even when they're far away.

**Built for the Build Small Hackathon (Hugging Face × Gradio, June 2026) — Thousand Token Wood track.**

## Source
Public GitHub repo: https://github.com/Sushruths04/moonlit-voice

This project was built with OpenAI Codex. See the commit history for Codex-attributed commits.

## How it works
1. A parent records 30–60s of their voice (or uploads a clip).
2. **`openbmb/VoxCPM2`** clones the voice.
3. The child picks a **genre** (animals, kingdom, space, dragons, ocean, forest) and **mood**
   (magical, funny, calming, dreamy), and optionally names the hero.
4. **`openbmb/MiniCPM5-1B`** writes a ~150-word, age-6, calming bedtime story.
5. VoxCPM2 narrates the story **in the parent's cloned voice**, ending on a gentle goodnight.
6. For **Kannada**: English story → **`ai4bharat/indictrans2-en-indic`** translation → **`ai4bharat/IndicF5`** narration in the parent's voice.

## Models & credits

### Core models
| Model | Purpose | Size | License |
|-------|---------|------|---------|
| `openbmb/VoxCPM2` | English voice cloning + narration | 2B | OpenBMB |
| `openbmb/MiniCPM5-1B` | Story generation | 1B | OpenBMB |
| `ai4bharat/IndicF5` | Kannada voice cloning + narration | 0.4B | MIT |
| `ai4bharat/indictrans2-en-indic-1B` | English → Kannada translation | 1B | MIT |

### Fine-tuned models (competition badges)

| Model | Purpose | Base | Training |
|-------|---------|------|----------|
| [`mitvho09/VoxCPM2-bedtime-lora`](https://huggingface.co/mitvho09/VoxCPM2-bedtime-lora) | English bedtime-story narration | `openbmb/VoxCPM2` | LoRA, LJSpeech, 100 steps |
| [`mitvho09/IndicF5-Kannada-Bedtime`](https://huggingface.co/mitvho09/IndicF5-Kannada-Bedtime) | Kannada bedtime-story narration | `ai4bharat/IndicF5` | Full fine-tune, Rasa Kannada, 300 steps |

### All models ≤ 32B ✅

## Fine-tuning details

### VoxCPM2 LoRA (English)
- **Dataset**: LJSpeech (13,100 clips, ~24h)
- **Method**: LoRA (r=32, alpha=64, DiT only)
- **Best experiment**: exp6 (r32, lr=5e-5, half data) — val loss 0.872
- **GPU**: A100-80GB (Modal)
- **Scripts**: `finetune/sweep.py`

### IndicF5 Kannada
- **Dataset**: Rasa Kannada (18 clips, ~2min)
- **Method**: Full fine-tune (CFM component)
- **Steps**: 300, final loss 0.4553
- **GPU**: A100-80GB (Modal)
- **Scripts**: `finetune/indicf5_finetune.py`

## Privacy
The parent's recording is processed in memory and **never stored on the server**. It is used only
to generate the narration for that session.

## Languages
- **English** — VoxCPM2 (stock + LoRA fine-tuned)
- **ಕನ್ನಡ Kannada** — IndicF5 (stock + fine-tuned) + IndicTrans2 translation

## Infrastructure
- **GPU inference**: Modal (A10G for English, A100-80GB for Kannada)
- **3 separate Modal images**: main (torch 2.11), IndicTrans2 (transformers 4.51.3), IndicF5 (torch 2.4.1)
- **Volumes**: `dreamvoice-ckpt` (checkpoints), `dreamvoice-ft-data` (training data)

## Links
- GitHub (Codex-built): `https://github.com/Sushruths04/moonlit-voice`
- Hugging Face Space: `<ADD SPACE URL>`
- VoxCPM2 LoRA: `https://huggingface.co/mitvho09/VoxCPM2-bedtime-lora`
- IndicF5 Kannada: `https://huggingface.co/mitvho09/IndicF5-Kannada-Bedtime`
- Demo video: `<ADD VIDEO LINK>`

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Env vars (only for stretch features): `HF_TOKEN`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
